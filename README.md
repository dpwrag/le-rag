# Le RAG

The RAG component of the project.

## Installation

Requirements:

- [uv](https://docs.astral.sh/uv/) package manager
- Podman or Docker

0. Clone this repository and `cd` to the project's root directory. 

### Using Google AI Model

1. Follow [Download and Run](https://qdrant.tech/documentation/quickstart/#download-and-run) step of Qdrant local quickstart guide to get Qdrant up and running.
1. [Set up Google API Key](https://ai.google.dev/gemini-api/docs/api-key).
1. `cp .env.example .env`.
1. Replace `QDRANT_URL` with your actual Qdrant URL from step 1.
1. Replace `GOOGLE_API_KEY` with your API key from step 2.
1. Set `MODEL_STRATEGY` to google
1. Install dependencies
    ```bash
    uv sync --dev
    ```

### Using Local AI Model

1. Follow [Download and Run](https://qdrant.tech/documentation/quickstart/#download-and-run) step of Qdrant local quickstart guide to get Qdrant up and running.
1. [Install Ollama](https://ollama.com/)
1. Pull the local AI model
    ```bash
    ollama pull hf.co/unsloth/Qwen3-1.7B-GGUF:Q4_K_M
    ```
1. `cp .env.example .env`.
1. Replace `QDRANT_URL` with your actual Qdrant URL from step 1.
1. Set `MODEL_STRATEGY` to local
1. Install dependencies
    ```bash
    uv sync --dev
    ```

## How to Run?

```bash
uv run --env-file .env main.py
```

Then, send a POST request to `http://localhost:8000/query` with the following JSON payload to get the prediction results.

```json
{
  "query": "string"
}
```

For more information, visit `http://localhost:8000/docs` (Swagger UI)

### Running DeepEval Evaluation

```bash
uv run --env-file .env python -m evaluation.deepeval_eval
```