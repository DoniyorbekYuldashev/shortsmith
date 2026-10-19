# Shortsmith

AI agents that turn one line of text into a finished ~30-second vertical short video:
research → script → storyboard → visuals + voice → edit → self-critique.

Work in progress. See [docs/PLAN.md](docs/PLAN.md) for the plan and
[docs/LEARNING_LOG.md](docs/LEARNING_LOG.md) for what was learned at each step.

## Setup

```bash
uv sync                  # create .venv and install the exact versions from uv.lock
cp .env.example .env     # then fill in your keys
```

Requires: [uv](https://docs.astral.sh/uv/), [ffmpeg](https://ffmpeg.org/).
