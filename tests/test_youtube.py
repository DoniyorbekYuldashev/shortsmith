"""research/youtube.py with a fake YouTube: httpx's MockTransport answers instead of Google.

The responses copy the shape of real ones (captured once, see the Phase 5 build notes); the
titles, channels and numbers are invented, so no YouTube data is stored in the repo.
"""

import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from shortsmith.research import youtube

NOW = datetime(2026, 10, 23, 20, 0, tzinfo=timezone.utc)  # 13:00 in California (PDT, UTC-7)


def video(vid: str, views: int, duration: str, *, likes: bool = True, width: str = "180", height: str = "320") -> dict:
    stats = {"viewCount": str(views), "favoriteCount": "0", "commentCount": "12"}
    if likes:
        stats["likeCount"] = str(views // 50)
    return {
        "id": vid,
        "snippet": {"publishedAt": "2026-10-04T05:45:30Z", "title": f"Title {vid}", "channelTitle": f"Channel {vid}"},
        "contentDetails": {"duration": duration},
        "statistics": stats,
        "player": {"embedHeight": height, "embedWidth": width},
    }


VIDEOS = [
    video("short1", 5_000, "PT39S"),
    video("long1", 900_000, "PT3M30S"),  # under 4 minutes (videoDuration=short), but longer than a Short
    video("short2", 120_000, "PT1M2S", likes=False, width="320", height="180"),  # likes hidden, horizontal
    video("live1", 50_000, "P0D"),  # a live stream: no length
]


class FakeYouTube:
    """Answers search.list and videos.list like the API, and remembers every request."""

    def __init__(self, ids: list[str] | None = None, videos: list[dict] | None = None, error: tuple[int, dict] | None = None):
        self.ids = [v["id"] for v in VIDEOS] if ids is None else ids
        self.videos = VIDEOS if videos is None else videos
        self.error = error
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.error:
            return httpx.Response(self.error[0], json=self.error[1])
        if request.url.path.endswith("/search"):
            return httpx.Response(200, json={"items": [{"id": {"videoId": i}} for i in self.ids]})
        return httpx.Response(200, json={"items": self.videos})

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "test-key-123")
    return {"cache_dir": tmp_path / "cache", "quota_path": tmp_path / "quota.jsonl"}


def search(fake: FakeYouTube, paths: dict, now: datetime = NOW, **kwargs) -> youtube.TrendReport:
    return youtube.search_trends("coffee shop Tashkent", "uz", client=fake.client(), now=now, **paths, **kwargs)


# --- durations ------------------------------------------------------------------------------


@pytest.mark.parametrize("iso, seconds", [
    ("PT39S", 39), ("PT1M2S", 62), ("PT3M", 180), ("PT1H0M5S", 3605), ("P1DT1S", 86401), ("P0D", 0),
    ("PT", None), ("P1W", None), ("nonsense", None),
])
def test_iso_8601_durations(iso, seconds):
    assert youtube.duration_seconds(iso) == seconds


# --- a live search ------------------------------------------------------------------------------


def test_only_shorts_length_videos_are_kept_most_viewed_first(paths):
    report = search(FakeYouTube(), paths)
    assert [v.video_id for v in report.videos] == ["short2", "short1"]  # long1 (210 s) and the live stream are out
    assert (report.candidates, report.shorts_length, report.units_used, report.from_cache) == (4, 2, 101, False)
    top = report.videos[0]
    assert (top.views, top.likes, top.comments, top.duration_s, top.vertical) == (120_000, None, 12, 62, False)
    assert report.videos[1].vertical is True and report.videos[1].likes == 100
    assert top.url == "https://www.youtube.com/watch?v=short2"


def test_the_request_asks_for_recent_short_videos_by_views_with_the_key_in_a_header(paths):
    fake = FakeYouTube()
    report = search(fake, paths)
    found, details = fake.requests
    params = found.url.params
    assert (params["type"], params["order"], params["videoDuration"]) == ("video", "viewCount", "short")
    assert (params["relevanceLanguage"], params["regionCode"]) == ("uz", "UZ")
    assert params["publishedAfter"] == "2026-09-23T20:00:00Z" == report.published_after.isoformat().replace("+00:00", "Z")
    assert details.url.params["id"] == "short1,long1,short2,live1"
    for request in fake.requests:
        assert request.headers["X-Goog-Api-Key"] == "test-key-123"
        assert "test-key-123" not in str(request.url)  # never in a URL: URLs end up in logs


def test_english_searches_have_no_region(paths):
    fake = FakeYouTube()
    youtube.search_trends("coffee", "en", client=fake.client(), now=NOW, **paths)
    assert "regionCode" not in fake.requests[0].url.params


def test_every_request_is_logged_with_its_units(paths):
    search(FakeYouTube(), paths)
    entries = [youtube.QuotaEntry.model_validate_json(line) for line in paths["quota_path"].read_text().splitlines()]
    assert [(e.endpoint, e.units, e.status) for e in entries] == [("search", 100, 200), ("videos", 1, 200)]
    assert youtube.units_today(paths["quota_path"], NOW) == 101


def test_no_results_means_no_videos_list_call(paths):
    fake = FakeYouTube(ids=[])
    report = search(fake, paths)
    assert len(fake.requests) == 1 and report.videos == [] and report.units_used == 100


# --- the cache --------------------------------------------------------------------------------


def test_a_repeated_search_comes_from_the_cache_and_costs_nothing(paths):
    search(FakeYouTube(), paths)
    again = FakeYouTube()
    report = youtube.search_trends("  Coffee shop   TASHKENT ", "uz", client=again.client(), now=NOW + timedelta(hours=23),
                                   **paths)  # the same search, typed differently
    assert again.requests == []
    assert (report.from_cache, report.units_used, len(report.videos)) == (True, 0, 2)
    assert youtube.units_today(paths["quota_path"], NOW) == 101


def test_the_cache_key_includes_the_language(paths):
    search(FakeYouTube(), paths)
    other = FakeYouTube()
    youtube.search_trends("coffee shop Tashkent", "ru", client=other.client(), now=NOW, **paths)
    assert len(other.requests) == 2


def test_an_entry_older_than_24_hours_is_deleted_and_searched_again(paths):
    search(FakeYouTube(), paths)
    [old_file] = paths["cache_dir"].glob("*.json")
    fresh = FakeYouTube()
    report = search(fresh, paths, now=NOW + timedelta(hours=25))
    assert len(fresh.requests) == 2 and not report.from_cache
    assert json.loads(old_file.read_text())["fetched_at"].startswith("2026-10-24T21:00")  # rewritten by the new search


def test_expired_entries_of_other_searches_are_purged_too(paths):
    youtube.search_trends("other topic", "en", client=FakeYouTube().client(), now=NOW, **paths)
    assert len(list(paths["cache_dir"].glob("*.json"))) == 1
    search(FakeYouTube(), paths, now=NOW + timedelta(days=2))
    assert len(list(paths["cache_dir"].glob("*.json"))) == 1  # only the new one: the API terms say don't keep data


# --- quota ------------------------------------------------------------------------------------


def test_the_quota_day_starts_at_midnight_pacific_time(paths):
    log = paths["quota_path"]
    youtube.record_units(youtube.QuotaEntry(ts=datetime(2026, 10, 23, 6, 30, tzinfo=timezone.utc), endpoint="search",
                                            units=100, query="q", status=200), log)  # 23:30 the day before, in California
    youtube.record_units(youtube.QuotaEntry(ts=datetime(2026, 10, 23, 7, 30, tzinfo=timezone.utc), endpoint="search",
                                            units=100, query="q", status=200), log)  # 00:30, today
    assert youtube.units_today(log, NOW) == 100


def test_a_search_over_our_daily_cap_is_refused_before_any_request(paths):
    youtube.record_units(youtube.QuotaEntry(ts=NOW, endpoint="search", units=1_950, query="q", status=200),
                         paths["quota_path"])
    fake = FakeYouTube()
    with pytest.raises(youtube.QuotaError, match="1950 of today's 2000"):
        search(fake, paths)
    assert fake.requests == []


def test_googles_quota_error_is_a_quota_error_and_its_units_are_still_counted(paths):
    error = {"error": {"code": 403, "message": "The request cannot be completed because you have exceeded your quota.",
                       "errors": [{"reason": "quotaExceeded", "domain": "youtube.quota"}]}}
    with pytest.raises(youtube.QuotaError, match="quotaExceeded"):
        search(FakeYouTube(error=(403, error)), paths)
    assert youtube.units_today(paths["quota_path"], NOW) == 100  # Google charges refused requests too


def test_other_api_errors_carry_the_apis_message_and_never_the_key(paths):
    error = {"error": {"code": 400, "message": "Invalid value at 'order'", "errors": [{"reason": "invalid"}]}}
    with pytest.raises(youtube.YouTubeError) as caught:
        search(FakeYouTube(error=(400, error)), paths)
    assert "HTTP 400 invalid: Invalid value at 'order'" in str(caught.value)
    assert "test-key-123" not in str(caught.value)
    assert not list(paths["cache_dir"].glob("*.json"))  # a failure is never cached


def test_a_missing_key_fails_before_any_request(paths, monkeypatch):
    monkeypatch.delenv("YOUTUBE_API_KEY")
    fake = FakeYouTube()
    with pytest.raises(youtube.YouTubeError, match="YOUTUBE_API_KEY is empty"):
        search(fake, paths)
    assert fake.requests == []
