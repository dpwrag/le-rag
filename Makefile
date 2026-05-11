.PHONY: dev ingest list-models

dev:
	uv run --env-file .env main.py

ingest:
	uv run --env-file .env python -m data.ingest_data

list-models:
	uv run --env-file .env list_models.py