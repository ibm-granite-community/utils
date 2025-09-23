# SPDX-License-Identifier: Apache-2.0

import pytest
from langchain_core.documents import Document
from transformers import AutoTokenizer, PreTrainedTokenizerBase


@pytest.fixture(scope="session")
def tokenizer() -> PreTrainedTokenizerBase:
    model_path = "ibm-granite/granite-3.3-8b-instruct"
    return AutoTokenizer.from_pretrained(model_path)


@pytest.fixture
def documents() -> list[Document]:
    docs = [
        Document(page_content="doc 49 text", metadata={"doc_id": 49}),
        Document(page_content="doc 12 text", metadata={"doc_id": 12}),
    ]
    return docs
