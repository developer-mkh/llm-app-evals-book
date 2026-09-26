import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

load_dotenv()

LM_STUDIO_MODEL = os.environ.get("LM_STUDIO_MODEL") or "google/gemma-4-12b-qat"
LM_STUDIO_BASE_URL = os.environ.get("LM_STUDIO_BASE_URL") or "http://host.docker.internal:1234/v1"
LM_STUDIO_API_KEY = os.environ.get("LM_STUDIO_API_KEY") or "lm-studio"


def get_model(*, thinking: bool = False) -> BaseChatModel:
    """Create a chat model instance using the LM Studio OpenAI-compatible API."""
    del thinking  # Keep compatibility with existing callers; LM Studio reasoning is not configured here.
    return init_chat_model(
        model=LM_STUDIO_MODEL,
        model_provider="openai",
        base_url=LM_STUDIO_BASE_URL,
        api_key=LM_STUDIO_API_KEY,
    )  # type: ignore[no-any-return]
