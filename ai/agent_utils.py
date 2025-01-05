# -*- coding: utf-8 -*-
import os

from config import (CODESTRAL_API_KEY,
                    LANGCHAIN_TRACING_V2,
                    HUGGINGFACE_API_KEY,
                    GOOGLE_API_KEY,
                    MISTRAL_API_KEY)

ASSISTANT_ENV = {
    "HUGGINGFACE_API_KEY": HUGGINGFACE_API_KEY,
    "GOOGLE_API_KEY": GOOGLE_API_KEY,
    "MISTRAL_API_KEY": MISTRAL_API_KEY,
    "CODESTRAL_API_KEY": CODESTRAL_API_KEY,
    "LANGCHAIN_TRACING_V2": LANGCHAIN_TRACING_V2,
}


def set_environment():
    diff = {key: value for key, value in ASSISTANT_ENV.items() if key not in os.environ}
    if len(diff) > 0:
        os.environ.update(diff)


