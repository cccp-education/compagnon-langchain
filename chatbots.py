# -*- coding: utf-8 -*-

import asyncio
import os

from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_mistralai.chat_models import ChatMistralAI

from config import (CODESTRAL_API_KEY,
                    LANGCHAIN_TRACING_V2,
                    HUGGINGFACE_API_KEY,
                    GOOGLE_API_KEY,
                    MISTRAL_API_KEY,
                    HUGGINGFACE_MODEL)

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


async def chabot(llm):
    chain = LLMChain(llm=llm, prompt=PromptTemplate(
        input_variables=["question"],
        template="Répondez à la question suivante : {question}"
    ))
    while True:
        print("To stop the chat: ':quit', ':exit', ':bye'")
        user_input = input("Vous: ")
        if user_input.lower() in [':quit', ':exit', ':bye']:
            print("Chatbot: Au revoir !")
            break
        response = await chain.ainvoke(input=user_input)
        print(f"Chatbot: {str(response['text']).strip()}")


async def ollama():
    await chabot(Ollama(
        model="smollm2:1.7b",
        # model="llama3.2:3b",
        # model="llama3.2:3b-instruct-q8_0",
        # model="dolphin3:8b",
        # model="dolphin3:8b-llama3.1-q8_0",
        base_url="http://localhost:11434",
        callback_manager=AsyncCallbackManager([StreamingStdOutCallbackHandler()]),
    ))


async def gemini_chat():
    await chabot(ChatGoogleGenerativeAI(
        api_key=GOOGLE_API_KEY,
        model="gemini-1.5-flash-002",
        temperature=0.7,
        callback_manager=AsyncCallbackManager([StreamingStdOutCallbackHandler()]),
    ))


async def huggingface_chat():
    await chabot(HuggingFaceEndpoint(
        huggingfacehub_api_token=HUGGINGFACE_API_KEY,
        model=HUGGINGFACE_MODEL,
        temperature=0.5,
        model_kwargs={"max_length": 1024},
        callback_manager=AsyncCallbackManager([StreamingStdOutCallbackHandler()]),
    ))


async def mistral_chat():
    await chabot(ChatMistralAI(
        # model_name="mistral-large-latest",
        # model_name="ministral-8b-latest",
        api_key=MISTRAL_API_KEY,
        model_name="ministral-8b-latest",
        temperature=0.7,
        max_retries=2,
        callback_manager=AsyncCallbackManager([StreamingStdOutCallbackHandler()]))
    )


if __name__ == '__main__':
    # asyncio.run(gemini_chat())
    # asyncio.run(mistral_chat())
    # asyncio.run(huggingface_chat())
    # asyncio.run(ollama())
