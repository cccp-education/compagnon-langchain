# -*- coding: utf-8 -*-

import asyncio

from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import Ollama

from config import HUGGINGFACE_API_KEY

async def ollama():
    # Initialiser Ollama
    llm = Ollama(
        base_url="http://localhost:11434",
        # model="dolphin-llama3:8b",
        model="llama3.2:3b",
        callback_manager=AsyncCallbackManager([StreamingStdOutCallbackHandler()]),
    )

    # Créer un template de prompt
    prompt = PromptTemplate(
        input_variables=["question"],
        template="Répondez à la question suivante : {question}"
    )

    # Créer une chaîne LLM
    chain = LLMChain(llm=llm, prompt=prompt)

    # Boucle principale du chatbot
    while True:
        user_input = input("Vous: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Chatbot: Au revoir !")
            break

        # Appel asynchrone à la chaîne
        response = await chain.arun(question=user_input)
        print(f"Chatbot: {response.strip()}")

async def huggingface_chat():
    # Initialiser le modèle Hugging Face
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-xxl",  # Vous pouvez changer le modèle selon vos besoins
        model_kwargs={"temperature": 0.5, "max_length": 512},
        huggingfacehub_api_token=HUGGINGFACE_API_KEY,
        callback_manager=AsyncCallbackManager([StreamingStdOutCallbackHandler()]),
    )

    # Créer un template de prompt
    prompt = PromptTemplate(
        input_variables=["question"],
        template="Répondez à la question suivante : {question}"
    )

    # Créer une chaîne LLM
    chain = LLMChain(llm=llm, prompt=prompt)

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
    asyncio.run(ollama())
    # asyncio.run(huggingface_chat())
