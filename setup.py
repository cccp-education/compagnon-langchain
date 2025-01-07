from setuptools import setup, find_packages

setup(
    name="compagnon-langchain",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "langchain-community>=0.3.7",
        "langchain-google-community>=2.0.0",
        "langchain-google-genai>=2.0.0",
        "langchain-huggingface>=0.1.0",
        "langchain-mistralai>=0.2.0",
        "langchain-ollama>=0.2.0",
        "langchain-postgres>=0.0.12",
        "langchain>=0.3.7",
        "assertpy>=1.1",
        "datasets>=3.2.0",
        "langchain-mistralai>=0.2.4",
        "langchain-groq>=0.2.2"
    ],
)
