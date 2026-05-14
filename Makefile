.PHONY: dev ingest list-models eval graph-only

dev:
	uv run --env-file .env main.py

ingest:
	uv run --env-file .env python -m data.ingest_data

list-models:
	uv run --env-file .env list_models.py
eval:
	uv run --env-file .env python -m evaluation.ragas_eval
graph-only:
	uv run --env-file .env graph_only.py