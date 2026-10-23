"""Real trend data: the most-viewed Shorts-length videos about a topic, from the last 30 days.

A live search is two requests to the YouTube Data API v3:
- search.list (100 units): the ids of videos that match the topic, ordered by views.
  It has no view counts and no exact lengths, and `videoDuration=short` only means "under 4 minutes".
- videos.list (1 unit for up to 50 ids): titles, channels, view counts and exact durations,
  so only videos of real Shorts length are kept.

The free quota is 10,000 units a day per Google Cloud project, reset at midnight Pacific time,
so one search is 1% of a day. Two guards: results are cached on disk for 24 hours (a repeated
topic costs nothing), and every request's units are appended to a quota log; a live search is
refused once today's units would pass DAILY_LIMIT_UNITS, our own cap below Google's.
YouTube's API terms allow keeping API data only briefly and never redistributing it: the cache
is gitignored, entries expire after 24 hours, and expired files are deleted.

Try it from the repo root (prints the top videos; a second run within 24 h is served from the cache):
    uv run --env-file .env python -m shortsmith.research.youtube "coffee shop Tashkent" --lang uz
"""

import argparse
import hashlib
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from pydantic import BaseModel, Field, computed_field

API = "https://www.googleapis.com/youtube/v3"
SEARCH_UNITS = 100  # search.list costs the same for 1 or 50 results
VIDEOS_UNITS = 1  # videos.list: 1 unit for up to 50 ids, whatever parts are asked for
GOOGLE_DAILY_UNITS = 10_000
DAILY_LIMIT_UNITS = 2_000  # our cap: 19 searches a day. A bug in a loop can't burn the whole day's quota
PACIFIC = ZoneInfo("America/Los_Angeles")  # YouTube's quota day (needs the tzdata package on Windows)

CACHE_DIR = Path(".cache/youtube")  # relative to the repo root, like runs/; gitignored
QUOTA_LOG = Path("runs/youtube_quota.jsonl")  # next to the cost ledger, same JSON Lines idea
CACHE_TTL = timedelta(hours=24)

LOOKBACK_DAYS = 30
SEARCH_RESULTS = 25  # one page of candidates: some are longer than a Short and get filtered out
SHORTS_MAX_S = 180  # a YouTube Short can be up to 3 minutes long (since October 2024)

# relevanceLanguage and regionCode are hints, not filters: results "most relevant to" a language,
# and videos "viewable in" a country. Uzbek and Russian topics are for viewers in Uzbekistan.
LOCALE: dict[str, tuple[str, str | None]] = {"en": ("en", None), "uz": ("uz", "UZ"), "ru": ("ru", "UZ")}


class YouTubeError(RuntimeError):
    """The API said no. The message is the API's own text; the key is in a header, never in it."""


class QuotaError(YouTubeError):
    """Our daily cap or Google's quota would be passed: no request was sent (or Google refused it)."""


class TrendVideo(BaseModel):
    video_id: str
    title: str
    channel: str
    views: int
    likes: int | None  # a channel can hide its like count
    comments: int | None  # ...and turn comments off
    published: datetime
    duration_s: int
    vertical: bool | None  # from the embed player's shape; None if YouTube didn't say

    @computed_field
    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


class TrendReport(BaseModel):
    """One search's result: what the Researcher reads."""

    query: str
    language: str
    published_after: datetime
    fetched_at: datetime
    candidates: int  # videos the search returned, before the length filter
    shorts_length: int  # how many of them are 3 minutes or shorter
    videos: list[TrendVideo]  # the most viewed of those, first
    from_cache: bool = False
    units_used: int = 0  # quota spent by this call: 0 when it came from the cache


class QuotaEntry(BaseModel):
    """One request to the API: one line of the quota log."""

    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    endpoint: str  # "search" or "videos"
    units: int
    query: str
    status: int  # HTTP status: refused requests are counted too (Google charges them)


# --- the quota log ------------------------------------------------------------------------


def record_units(entry: QuotaEntry, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(entry.model_dump_json() + "\n")


def units_today(path: Path = QUOTA_LOG, now: datetime | None = None) -> int:
    """Units spent since midnight Pacific time, when Google resets the quota."""
    if not path.exists():
        return 0
    today = (now or datetime.now(timezone.utc)).astimezone(PACIFIC).date()
    entries = (QuotaEntry.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return sum(e.units for e in entries if e.ts.astimezone(PACIFIC).date() == today)


# --- the cache ----------------------------------------------------------------------------


def cache_file(cache_dir: Path, query: str, language: str, max_results: int) -> Path:
    """One file per search. The name is a hash: no topic text in file names, any language works."""
    key = f"{' '.join(query.lower().split())}|{language}|{max_results}|{LOOKBACK_DAYS}"
    return cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()[:24]}.json"


def read_cache(path: Path, now: datetime) -> TrendReport | None:
    """A fresh cached report, or None. A stale one is deleted (the API terms: don't keep it)."""
    if not path.exists():
        return None
    report = TrendReport.model_validate_json(path.read_text(encoding="utf-8"))
    if now - report.fetched_at > CACHE_TTL:
        path.unlink()
        return None
    return report.model_copy(update={"from_cache": True, "units_used": 0})


def purge_cache(cache_dir: Path, now: datetime) -> int:
    """Delete every expired entry, not just the one asked for. Returns how many went."""
    removed = 0
    for path in cache_dir.glob("*.json"):
        try:
            stale = now - TrendReport.model_validate_json(path.read_text(encoding="utf-8")).fetched_at > CACHE_TTL
        except ValueError:
            stale = True  # unreadable: an old format or a half-written file
        if stale:
            path.unlink()
            removed += 1
    return removed


# --- the two API calls ----------------------------------------------------------------------


_DURATION = re.compile(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$")


def duration_seconds(iso: str) -> int | None:
    """YouTube's ISO 8601 length ("PT1M5S") in seconds; None for anything else (e.g. "P1W")."""
    match = _DURATION.match(iso)
    if not match or iso in ("P", "PT"):
        return None
    days, hours, minutes, seconds = (int(part or 0) for part in match.groups())
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def _get(client: httpx.Client, endpoint: str, params: dict[str, Any], units: int, query: str, quota_path: Path) -> dict:
    """One GET, its units logged whatever the answer. The key travels in a header, so it is in
    no URL, log line or error message."""
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        raise YouTubeError("YOUTUBE_API_KEY is empty: add it to .env")
    response = client.get(f"{API}/{endpoint}", params=params, headers={"X-Goog-Api-Key": key})
    record_units(QuotaEntry(endpoint=endpoint, units=units, query=query, status=response.status_code), quota_path)
    if response.status_code == 200:
        return response.json()
    try:
        error = response.json()["error"]
        reasons = {e.get("reason") for e in error.get("errors", [])}
        message = f"{endpoint}: HTTP {response.status_code} {', '.join(sorted(r for r in reasons if r))}: {error.get('message', '')}"
    except (ValueError, KeyError, TypeError):
        reasons, message = set(), f"{endpoint}: HTTP {response.status_code}"
    if reasons & {"quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded"}:
        raise QuotaError(message)
    raise YouTubeError(message)


def _video(item: dict) -> TrendVideo | None:
    """One videos.list item, or None if it isn't a finished video with a known length."""
    duration = duration_seconds(item["contentDetails"]["duration"])
    stats = item.get("statistics", {})
    if duration is None or "viewCount" not in stats:
        return None
    player = item.get("player", {})
    width, height = player.get("embedWidth"), player.get("embedHeight")
    return TrendVideo(
        video_id=item["id"],
        title=item["snippet"]["title"],
        channel=item["snippet"]["channelTitle"],
        views=int(stats["viewCount"]),
        likes=int(stats["likeCount"]) if "likeCount" in stats else None,
        comments=int(stats["commentCount"]) if "commentCount" in stats else None,
        published=item["snippet"]["publishedAt"],
        duration_s=duration,
        vertical=int(height) > int(width) if width and height else None,
    )


def search_trends(
    query: str,
    language: str = "en",
    max_results: int = 10,
    *,
    client: httpx.Client | None = None,
    now: datetime | None = None,
    cache_dir: Path = CACHE_DIR,
    quota_path: Path = QUOTA_LOG,
    daily_limit: int = DAILY_LIMIT_UNITS,
) -> TrendReport:
    """The most-viewed Shorts-length videos for `query` from the last 30 days, most viewed first.

    Served from the cache when the same search ran in the last 24 hours (0 units); otherwise
    101 units, refused with QuotaError if that would pass today's cap.
    """
    if language not in LOCALE:
        raise ValueError(f"language {language!r}, expected one of {sorted(LOCALE)}")
    now = now or datetime.now(timezone.utc)
    purge_cache(cache_dir, now)
    path = cache_file(cache_dir, query, language, max_results)
    if (cached := read_cache(path, now)) is not None:
        return cached

    spent = units_today(quota_path, now)
    if spent + SEARCH_UNITS + VIDEOS_UNITS > daily_limit:
        raise QuotaError(f"a live search needs {SEARCH_UNITS + VIDEOS_UNITS} units; {spent} of today's "
                         f"{daily_limit} are used (our cap; Google's is {GOOGLE_DAILY_UNITS}). Try again after midnight Pacific time.")

    relevance_language, region = LOCALE[language]
    published_after = (now - timedelta(days=LOOKBACK_DAYS)).replace(microsecond=0)
    search_params = {
        "part": "id", "type": "video", "q": query, "order": "viewCount", "videoDuration": "short",
        "publishedAfter": published_after.isoformat().replace("+00:00", "Z"), "maxResults": SEARCH_RESULTS,
        "relevanceLanguage": relevance_language, "safeSearch": "moderate",
        "fields": "items(id(videoId))",  # a smaller answer; the quota cost is the same
    }
    if region:
        search_params["regionCode"] = region

    own_client = client is None
    client = client or httpx.Client(timeout=20)
    try:
        found = _get(client, "search", search_params, SEARCH_UNITS, query, quota_path)
        ids = [item["id"]["videoId"] for item in found.get("items", []) if item.get("id", {}).get("videoId")]
        items = []
        if ids:
            details = _get(client, "videos", {
                "part": "snippet,statistics,contentDetails,player", "id": ",".join(ids),
                "maxWidth": 180,  # asks for the embed player's size, which follows the video's shape
                "fields": "items(id,snippet(title,channelTitle,publishedAt),statistics,contentDetails(duration),player(embedWidth,embedHeight))",
            }, VIDEOS_UNITS, query, quota_path)
            items = details.get("items", [])
    finally:
        if own_client:
            client.close()

    videos = [v for item in items if (v := _video(item)) and 0 < v.duration_s <= SHORTS_MAX_S]
    videos.sort(key=lambda v: v.views, reverse=True)
    report = TrendReport(query=query, language=language, published_after=published_after, fetched_at=now,
                         candidates=len(ids), shorts_length=len(videos), videos=videos[:max_results])
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report.model_copy(update={"units_used": SEARCH_UNITS + (VIDEOS_UNITS if ids else 0)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Top Shorts-length videos for a topic, last 30 days (YouTube Data API).")
    parser.add_argument("query")
    parser.add_argument("--lang", choices=sorted(LOCALE), default="en")
    parser.add_argument("-n", type=int, default=10, help="how many videos to show")
    args = parser.parse_args()
    report = search_trends(args.query, args.lang, args.n)
    source = "cache (0 units)" if report.from_cache else f"live ({report.units_used} units)"
    print(f"{args.query!r} ({args.lang}) | since {report.published_after:%Y-%m-%d} | {source} | "
          f"{report.candidates} candidates, {report.shorts_length} of them Shorts-length, top {len(report.videos)}:\n")
    for v in report.videos:
        shape = {True: "vertical", False: "horizontal", None: "?"}[v.vertical]
        print(f"{v.views:>12,} views | {v.duration_s:>3} s | {shape:<10} | {v.published:%Y-%m-%d} | {v.channel[:24]:<24} | {v.title}")
    print(f"\nquota: {units_today()} units used today (Pacific time) of our {DAILY_LIMIT_UNITS} cap / Google's {GOOGLE_DAILY_UNITS}")


if __name__ == "__main__":
    main()
