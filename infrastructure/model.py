import os
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel
from typing import Tuple, Callable
from langchain_google_genai import ChatGoogleGenerativeAI

LOCAL_MODEL = "hf.co/unsloth/Qwen3-1.7B-GGUF:Q4_K_M"
GOOGLE_MODEL = "gemma-4-31b-it"


_strategies: dict[str, Callable] = {}


def _register_strategy(name: str) -> Callable:
    def decorator(func: Callable):
        key = name
        _strategies[key] = func
        return func

    return decorator


def create_model() -> Tuple[BaseChatModel, str]:
    strategy = os.environ.get("MODEL_STRATEGY", "local")
    return _strategies[strategy]()


@_register_strategy("local")
def _create_local_model() -> Tuple[ChatOllama, str]:
    return ChatOllama(
        model=LOCAL_MODEL,
        temperature=1.0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    ), LOCAL_MODEL


@_register_strategy("google")
def _create_google_model() -> Tuple[ChatGoogleGenerativeAI, str]:
    return ChatGoogleGenerativeAI(
        model=GOOGLE_MODEL,
        temperature=1.0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    ), GOOGLE_MODEL
