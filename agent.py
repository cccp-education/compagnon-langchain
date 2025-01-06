# -*- coding: utf-8 -*-

import asyncio
import os

from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI

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


async def ollama():
    chain = LLMChain(llm=Ollama(
        base_url="http://localhost:11434",
        # model="dolphin-llama3:8b",
        model="llama3.2:3b",
        callback_manager=AsyncCallbackManager([StreamingStdOutCallbackHandler()]),
    ), prompt=PromptTemplate(
        input_variables=["question"],
        template="Répondez à la question suivante : {question}"
    ))
    while True:
        user_input = input("Vous: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Chatbot: Au revoir !")
            break
        response = await chain.arun(question=user_input)
        print(f"Chatbot: {response.strip()}")


async def gemini_chat():
    chain = LLMChain(
        llm=ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-002",
            temperature=0.7,
            google_api_key=GOOGLE_API_KEY,
            callback_manager=AsyncCallbackManager([StreamingStdOutCallbackHandler()]),
        ),
        prompt=PromptTemplate(
            input_variables=["question"],
            template="Répondez à la question suivante : {question}"
        )
    )
    while True:
        user_input = input("Vous: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Chatbot: Au revoir !")
            break
        response = await chain.arun(question=user_input)
        print(f"Chatbot: {response.strip()}")

async def huggingface_chat():
    # Créer une chaîne LLM
    chain = LLMChain(llm=HuggingFaceEndpoint(
        model=HUGGINGFACE_MODEL,
        temperature=0.5,
        model_kwargs={"max_length": 1024},
        huggingfacehub_api_token=HUGGINGFACE_API_KEY,
        callback_manager=AsyncCallbackManager([StreamingStdOutCallbackHandler()]),
    ), prompt=PromptTemplate(
        input_variables=["question"],
        template="Répondez à la question suivante : {question}"
    ))

    # Boucle principale du chatbot
    while True:
        user_input = input("Vous: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Chatbot: Au revoir !")
            break

        # Appel asynchrone à la chaîne
        response = await chain.arun(question=user_input)
        print(f"Chatbot: {response.strip()}")


if __name__ == '__main__':
    # asyncio.run(ollama())
    # asyncio.run(huggingface_chat())
    asyncio.run(gemini_chat())
