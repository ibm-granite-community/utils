# SPDX-License-Identifier: Apache-2.0

# create_stuff_documents_chain tests

import json
from functools import partial
from typing import Any

import pytest
from assertpy import assert_that
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, convert_to_openai_messages
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from transformers import PreTrainedTokenizerBase

from ibm_granite_community.langchain.chains.combine_documents import create_stuff_documents_chain
from ibm_granite_community.langchain.prompts import TokenizerChatPromptTemplate


# Method to use as a tool
def i_am_a_tool(tool_arg: str) -> str:
    """I am a tool!

    Args:
        tool_arg: The tool argument

    Returns:
        str: The tool argument.
    """
    return tool_arg


def identity_llm(input: LanguageModelInput, **kwargs: Any) -> str:  # pylint: disable=redefined-builtin
    """Mock llm which returns the formatted prompt, messages and kwargs in its output"""
    if not isinstance(input, PromptValue):
        raise ValueError
    # Use client-side prompt formatting
    prompt = input.to_string()
    result = json.dumps(dict(kwargs, prompt=prompt, messages=[repr(message) for message in input.to_messages()]))
    return result


def identity_chat_llm(tokenizer: PreTrainedTokenizerBase, input: LanguageModelInput, **kwargs: Any) -> BaseMessage:  # pylint: disable=redefined-builtin
    """Mock chat llm which returns the formatted prompt, messages and kwargs in its output"""
    if not isinstance(input, PromptValue):
        raise ValueError
    # Emulate server-side prompt formatting (don't call input.to_string())
    conversation = convert_to_openai_messages(input.to_messages())
    if not isinstance(conversation, list):
        conversation = [conversation]
    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,  # output is str
        add_generation_prompt=True,
        **kwargs,
    )
    result = json.dumps(dict(kwargs, prompt=prompt, messages=[repr(message) for message in input.to_messages()]))
    return AIMessage(result)


class TestDocumentsChain:
    @pytest.mark.parametrize("document_variable_name", ["context", "custom_name"])
    def test_documents_chain(self, tokenizer, document_variable_name):
        assert_that(tokenizer).is_not_none()
        llm = RunnableLambda(identity_llm)
        prompt_template = TokenizerChatPromptTemplate.from_template(
            "user content",
            tokenizer=tokenizer,
        )
        chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template, document_variable_name=document_variable_name, output_parser=JsonOutputParser())
        documents = [
            Document(page_content="doc 49 text", metadata={"doc_id": 49}),
            Document(page_content="doc 12 text", metadata={"doc_id": 12}),
        ]
        result = chain.invoke(input={document_variable_name: documents})
        (
            assert_that(result["prompt"])
            .matches(r"(?ms)<\|start_of_role\|>user<\|end_of_role\|>\s*?user content\s*?<\|end_of_text\|>")
            .contains(*(document.page_content for document in documents))
            .ends_with("<|start_of_role|>assistant<|end_of_role|>")
        )
        assert_that(result["messages"]).is_length(1)
        assert_that(result["messages"][0]).contains("user content")
        assert_that(result["documents"]).extracting("text").contains(*(document.page_content for document in documents))
        assert_that(result["documents"]).extracting("doc_id").contains(*(document.metadata["doc_id"] for document in documents))

    @pytest.mark.parametrize("document_variable_name", ["context", "custom_name"])
    def test_documents_chain_chat(self, tokenizer, document_variable_name):
        assert_that(tokenizer).is_not_none()
        llm = RunnableLambda(partial(identity_chat_llm, tokenizer))
        prompt_template = ChatPromptTemplate.from_template(
            "user content",
        )
        chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template, document_variable_name=document_variable_name, output_parser=JsonOutputParser())
        documents = [
            Document(page_content="doc 49 text", metadata={"doc_id": 49}),
            Document(page_content="doc 12 text", metadata={"doc_id": 12}),
        ]
        result = chain.invoke(input={document_variable_name: documents})
        (
            assert_that(result["prompt"])
            .matches(r"(?ms)<\|start_of_role\|>user<\|end_of_role\|>\s*?user content\s*?<\|end_of_text\|>")
            .contains(*(document.page_content for document in documents))
            .ends_with("<|start_of_role|>assistant<|end_of_role|>")
        )
        assert_that(result["messages"]).is_length(1)
        assert_that(result["messages"][0]).contains("user content")
        assert_that(result["documents"]).extracting("text").contains(*(document.page_content for document in documents))
        assert_that(result["documents"]).extracting("doc_id").contains(*(document.metadata["doc_id"] for document in documents))

    def test_documents_chain_bind(self, tokenizer):
        assert_that(tokenizer).is_not_none()
        tools = [convert_to_openai_tool(i_am_a_tool)]
        llm = RunnableLambda(identity_llm).bind(tools=tools)
        prompt_template = TokenizerChatPromptTemplate.from_messages(
            messages=[
                MessagesPlaceholder("user_content"),
            ],
            tokenizer=tokenizer,
        ).bind(
            user_content=[
                HumanMessage(content="user content"),
            ]
        )
        chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template, output_parser=JsonOutputParser())
        documents = [
            Document(page_content="doc 49 text", metadata={"doc_id": 49}),
            Document(page_content="doc 12 text", metadata={"doc_id": 12}),
        ]
        result = chain.invoke(input={"context": documents})
        (
            assert_that(result["prompt"])
            .matches(r"(?ms)<\|start_of_role\|>user<\|end_of_role\|>\s*?user content\s*?<\|end_of_text\|>")
            .contains(*(document.page_content for document in documents))
            .ends_with("<|start_of_role|>assistant<|end_of_role|>")
        )
        assert_that(result["messages"]).is_length(1)
        assert_that(result["messages"][0]).contains("user content")
        assert_that(result["documents"]).extracting("text").contains(*(document.page_content for document in documents))
        assert_that(result["documents"]).extracting("doc_id").contains(*(document.metadata["doc_id"] for document in documents))
        assert_that(result["tools"]).is_length(1)
        assert_that(result["tools"]).extracting("type").contains_only(tools[0]["type"])
        assert_that(result["tools"]).extracting("function").extracting("name").contains_only(tools[0]["function"]["name"])

    @pytest.mark.asyncio
    @pytest.mark.parametrize("document_variable_name", ["context", "custom_name"])
    async def test_documents_chain_async(self, tokenizer, document_variable_name):
        assert_that(tokenizer).is_not_none()
        llm = RunnableLambda(identity_llm)
        prompt_template = TokenizerChatPromptTemplate.from_template(
            "user content",
            tokenizer=tokenizer,
        )
        chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template, document_variable_name=document_variable_name, output_parser=JsonOutputParser())
        documents = [
            Document(page_content="doc 49 text", metadata={"doc_id": 49}),
            Document(page_content="doc 12 text", metadata={"doc_id": 12}),
        ]
        result = await chain.ainvoke(input={document_variable_name: documents})
        (
            assert_that(result["prompt"])
            .matches(r"(?ms)<\|start_of_role\|>user<\|end_of_role\|>\s*?user content\s*?<\|end_of_text\|>")
            .contains(*(document.page_content for document in documents))
            .ends_with("<|start_of_role|>assistant<|end_of_role|>")
        )
        assert_that(result["messages"]).is_length(1)
        assert_that(result["messages"][0]).contains("user content")
        assert_that(result["documents"]).extracting("text").contains(*(document.page_content for document in documents))
        assert_that(result["documents"]).extracting("doc_id").contains(*(document.metadata["doc_id"] for document in documents))

    @pytest.mark.asyncio
    @pytest.mark.parametrize("document_variable_name", ["context", "custom_name"])
    async def test_documents_chain_chat_async(self, tokenizer, document_variable_name):
        assert_that(tokenizer).is_not_none()
        llm = RunnableLambda(partial(identity_chat_llm, tokenizer))
        prompt_template = ChatPromptTemplate.from_template(
            "user content",
        )
        chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template, document_variable_name=document_variable_name, output_parser=JsonOutputParser())
        documents = [
            Document(page_content="doc 49 text", metadata={"doc_id": 49}),
            Document(page_content="doc 12 text", metadata={"doc_id": 12}),
        ]
        result = await chain.ainvoke(input={document_variable_name: documents})
        (
            assert_that(result["prompt"])
            .matches(r"(?ms)<\|start_of_role\|>user<\|end_of_role\|>\s*?user content\s*?<\|end_of_text\|>")
            .contains(*(document.page_content for document in documents))
            .ends_with("<|start_of_role|>assistant<|end_of_role|>")
        )
        assert_that(result["messages"]).is_length(1)
        assert_that(result["messages"][0]).contains("user content")
        assert_that(result["documents"]).extracting("text").contains(*(document.page_content for document in documents))
        assert_that(result["documents"]).extracting("doc_id").contains(*(document.metadata["doc_id"] for document in documents))

    def test_documents_chain_bind_chat(self, tokenizer):
        assert_that(tokenizer).is_not_none()
        tools = [convert_to_openai_tool(i_am_a_tool)]
        llm = RunnableLambda(partial(identity_chat_llm, tokenizer)).bind(tools=tools)
        prompt_template = ChatPromptTemplate.from_messages(
            messages=[
                MessagesPlaceholder("user_content"),
            ],
        )
        chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template, output_parser=JsonOutputParser())
        documents = [
            Document(page_content="doc 49 text", metadata={"doc_id": 49}),
            Document(page_content="doc 12 text", metadata={"doc_id": 12}),
        ]
        result = chain.invoke(
            input={
                "context": documents,
                "user_content": [
                    HumanMessage(content="user content"),
                ],
            }
        )
        (
            assert_that(result["prompt"])
            .matches(r"(?ms)<\|start_of_role\|>user<\|end_of_role\|>\s*?user content\s*?<\|end_of_text\|>")
            .contains(*(document.page_content for document in documents))
            .ends_with("<|start_of_role|>assistant<|end_of_role|>")
        )
        assert_that(result["messages"]).is_length(1)
        assert_that(result["messages"][0]).contains("user content")
        assert_that(result["documents"]).extracting("text").contains(*(document.page_content for document in documents))
        assert_that(result["documents"]).extracting("doc_id").contains(*(document.metadata["doc_id"] for document in documents))
        assert_that(result["tools"]).is_length(1)
        assert_that(result["tools"]).extracting("type").contains_only(tools[0]["type"])
        assert_that(result["tools"]).extracting("function").extracting("name").contains_only(tools[0]["function"]["name"])
