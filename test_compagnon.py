# -*- coding: utf-8 -*-
import os
from unittest import TestCase, main

from assertpy import assert_that
from datasets import load_dataset
from langchain_google_genai import GoogleGenerativeAI

from agent import ASSISTANT_ENV, set_environment
from config import GOOGLE_API_KEY
from dataset import format_dataset_to_json


class CompagnonTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        set_environment()
        for key, value in ASSISTANT_ENV.items():
            assert_that(os.environ[key]).is_equal_to(value)

    @staticmethod
    def test_dataset():
        dataset = load_dataset("imdb")
        assert_that(format_dataset_to_json(
            dataset,
            100, keys=["text", "label"])
        ).is_not_empty()

    @staticmethod
    def test_gemini_chat_connection():
        if GOOGLE_API_KEY not in os.environ:
            set_environment()
        assert_that(GOOGLE_API_KEY).is_not_empty()
        assert_that(ASSISTANT_ENV).contains_key("GOOGLE_API_KEY")
        GoogleGenerativeAI(model="gemini-1.5", temperature=0.7)


if __name__ == '__main__':
    main()
