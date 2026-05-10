.PHONY: dev ingest

dev:
	uv run --env-file .env main.py

ingest:
	uv run --env-file .env python -m data.ingest_data