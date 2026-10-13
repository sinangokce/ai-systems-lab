uv init
uv sync
uv add torch
uv add numpy

uv run ruff check .
uv run ruff format .
uv run pytest