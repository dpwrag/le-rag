# Le RAG

The RAG component of the project.

## Installation

Requirements:

- [uv](https://docs.astral.sh/uv/) package manager
- Podman or Docker

Installation steps:

1. [Set up the API Key](https://ai.google.dev/gemini-api/docs/api-key).
1. Follow [Download and Run](https://qdrant.tech/documentation/quickstart/#download-and-run) step of Qdrant local quickstart guide to get Qdrant up and running.
1. `cp .env.example .env`.
1. Replace `GOOGLE_API_KEY` with your actual API key.
1. Replace `QDRANT_URL` with your actual Qdrant URL from step 2.
1. Installl dependencies
    ```bash
    uv sync
    ```
1. Run `main.py`
    ```bash
    uv run --env-file .env main.py
    ```

## Related Docs
- [Qdrant docs](https://python-client.qdrant.tech/qdrant_client.qdrant_client)