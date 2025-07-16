# SPDX-License-Identifier: Apache-2.0

# TokenizerChatPromptTemplate tests

import json
import re

import pytest
from assertpy import assert_that
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, ChatMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

from ibm_granite_community.langchain import TokenizerChatPromptTemplate, create_stuff_documents_chain


# Method to use as a tool
def i_am_a_tool(tool_arg: str) -> str:
    """I am a tool!

    Args:
        tool_arg: The tool argument

    Returns:
        str: The tool argument.
    """
    return tool_arg


class TestTokenizerChatTemplate:
    # Test simple from_template prompt
    def test_from_template(self, tokenizer):
        assert_that(tokenizer).is_not_none()
        prompt_template = TokenizerChatPromptTemplate.from_template("What's the weather like in Bengaluru?", tokenizer=tokenizer)
        text = prompt_template.invoke(input={}).to_string()
        (
            assert_that(text)
            .contains("<|start_of_role|>user<|end_of_role|>What's the weather like in Bengaluru?<|end_of_text|>")
            .ends_with("<|start_of_role|>assistant<|end_of_role|>")
        )

    @pytest.mark.asyncio
    async def test_from_template_async(self, tokenizer):
        assert_that(tokenizer).is_not_none()
        prompt_template = TokenizerChatPromptTemplate.from_template("What's the weather like in Bengaluru?", tokenizer=tokenizer)
        text = (await prompt_template.ainvoke(input={})).to_string()
        (
            assert_that(text)
            .contains("<|start_of_role|>user<|end_of_role|>What's the weather like in Bengaluru?<|end_of_text|>")
            .ends_with("<|start_of_role|>assistant<|end_of_role|>")
        )

    def test_from_messages(self, tokenizer):
        assert_that(tokenizer).is_not_none()
        prompt_template = TokenizerChatPromptTemplate.from_messages(
            [
                SystemMessage(content="system content"),
                HumanMessage(content="user content1\nuser content2\n"),
                ChatMessage(role="User", content="chat content"),
                AIMessage(content="assistant content"),
            ],
            tokenizer=tokenizer,
        )
        text = prompt_template.invoke(input={}).to_string()
        (
            assert_that(text)
            .matches(r"(?ms)<\|start_of_role\|>system<\|end_of_role\|>\s*?system content\s*?<\|end_of_text\|>")
            .matches(r"(?ms)<\|start_of_role\|>user<\|end_of_role\|>\s*?user content1\s*?user content2\s*?<\|end_of_text\|>")
            .matches(r"(?ms)<\|start_of_role\|>user<\|end_of_role\|>\s*?chat content\s*?<\|end_of_text\|>")
            .matches(r"(?ms)<\|start_of_role\|>assistant<\|end_of_role\|>\s*?assistant content\s*?<\|end_of_text\|>")
            .ends_with("<|start_of_role|>assistant<|end_of_role|>")
        )

    @pytest.mark.asyncio
    async def test_from_messages_async(self, tokenizer):
        assert_that(tokenizer).is_not_none()
        prompt_template = TokenizerChatPromptTemplate.from_messages(
            [
                SystemMessage(content="system content"),
                HumanMessage(content="user content1\nuser content2\n"),
                ChatMessage(role="User", content="chat content"),
                AIMessage(content="assistant content"),
            ],
            tokenizer=tokenizer,
        )
        text = (await prompt_template.ainvoke(input={})).to_string()
        (
            assert_that(text)
            .matches(r"(?ms)<\|start_of_role\|>system<\|end_of_role\|>\s*?system content\s*?<\|end_of_text\|>")
            .matches(r"(?ms)<\|start_of_role\|>user<\|end_of_role\|>\s*?user content1\s*?user content2\s*?<\|end_of_text\|>")
            .matches(r"(?ms)<\|start_of_role\|>user<\|end_of_role\|>\s*?chat content\s*?<\|end_of_text\|>")
            .matches(r"(?ms)<\|start_of_role\|>assistant<\|end_of_role\|>\s*?assistant content\s*?<\|end_of_text\|>")
            .ends_with("<|start_of_role|>assistant<|end_of_role|>")
        )

    def test_bind(self, tokenizer):
        assert_that(tokenizer).is_not_none()
        prompt_template = TokenizerChatPromptTemplate.from_messages(
            [
                SystemMessage(content="system content"),
                HumanMessage(content="user content"),
                ChatMessage(role="User", content="chat content"),
                AIMessage(content="assistant content"),
                MessagesPlaceholder("tool_results"),
            ],
            tokenizer=tokenizer,
        ).bind(tools=[i_am_a_tool])
        text = prompt_template.invoke(
            input={
                "tool_results": [
                    ToolMessage(tool_call_id="call_id_1", content='{"name": "tool1", "arguments": {"a": "b"}}'),
                ]
            }
        ).to_string()
        (
            assert_that(text)
            .matches(r"(?ms)<\|start_of_role\|>system<\|end_of_role\|>\s*?system content\s*?<\|end_of_text\|>")
            .matches(r"(?ms)<\|start_of_role\|>user<\|end_of_role\|>\s*?user content\s*?<\|end_of_text\|>")
            .matches(r"(?ms)<\|start_of_role\|>user<\|end_of_role\|>\s*?chat content\s*?<\|end_of_text\|>")
            .matches(r"(?ms)<\|start_of_role\|>assistant<\|end_of_role\|>\s*?assistant content\s*?<\|end_of_text\|>")
            .ends_with("<|start_of_role|>assistant<|end_of_role|>")
        )

        match = re.search(r"(?ms)<\|start_of_role\|>(tools|available_tools)<\|end_of_role\|>(?P<tools>.*?)<\|end_of_text\|>", text)
        assert_that(match).is_not_none()
        if match:
            tools = match.group("tools")
            assert_that(tools).is_not_none()
            parsed = json.loads(tools.strip())
            assert_that(parsed).is_length(1)
            tool = parsed[0]
            assert_that(tool).contains_entry({"type": "function"}).contains_key("function")
            assert_that(tool["function"]).contains_entry({"name": "i_am_a_tool"}).contains_key("description", "parameters")

        match = re.search(r"(?ms)<\|start_of_role\|>tool<\|end_of_role\|>(?P<tool>.*?)<\|end_of_text\|>", text)
        assert_that(match).is_not_none()
        if match:
            tool = match.group("tool")
            assert_that(tool).is_not_none()
            parsed = json.loads(tool.strip())
            assert_that(parsed).contains_entry({"name": "tool1"}).contains_key("arguments")

    @pytest.mark.asyncio
    async def test_bind_async(self, tokenizer):
        assert_that(tokenizer).is_not_none()
        prompt_template = TokenizerChatPromptTemplate.from_messages(
            [
                SystemMessage(content="system content"),
                HumanMessage(content="user content"),
                ChatMessage(role="User", content="chat content"),
                AIMessage(content="assistant content"),
                MessagesPlaceholder("tool_results"),
            ],
            tokenizer=tokenizer,
        ).bind(tools=[i_am_a_tool])
        text = (
            await prompt_template.ainvoke(
                input={
                    "tool_results": [
                        ToolMessage(tool_call_id="call_id_1", content='{"name": "tool1", "arguments": {"a": "b"}}'),
                    ]
                }
            )
        ).to_string()
        (
            assert_that(text)
            .matches(r"(?ms)<\|start_of_role\|>system<\|end_of_role\|>\s*?system content\s*?<\|end_of_text\|>")
            .matches(r"(?ms)<\|start_of_role\|>user<\|end_of_role\|>\s*?user content\s*?<\|end_of_text\|>")
            .matches(r"(?ms)<\|start_of_role\|>user<\|end_of_role\|>\s*?chat content\s*?<\|end_of_text\|>")
            .matches(r"(?ms)<\|start_of_role\|>assistant<\|end_of_role\|>\s*?assistant content\s*?<\|end_of_text\|>")
            .ends_with("<|start_of_role|>assistant<|end_of_role|>")
        )

        match = re.search(r"(?ms)<\|start_of_role\|>(tools|available_tools)<\|end_of_role\|>(?P<tools>.*?)<\|end_of_text\|>", text)
        assert_that(match).is_not_none()
        if match:
            tools = match.group("tools")
            assert_that(tools).is_not_none()
            parsed = json.loads(tools.strip())
            assert_that(parsed).is_length(1)
            tool = parsed[0]
            assert_that(tool).contains_entry({"type": "function"}).contains_key("function")
            assert_that(tool["function"]).contains_entry({"name": "i_am_a_tool"}).contains_key("description", "parameters")

        match = re.search(r"(?ms)<\|start_of_role\|>tool<\|end_of_role\|>(?P<tool>.*?)<\|end_of_text\|>", text)
        assert_that(match).is_not_none()
        if match:
            tool = match.group("tool")
            assert_that(tool).is_not_none()
            parsed = json.loads(tool.strip())
            assert_that(parsed).contains_entry({"name": "tool1"}).contains_key("arguments")

    def test_documents(self, tokenizer):
        assert_that(tokenizer).is_not_none()
        prompt_template = TokenizerChatPromptTemplate.from_template(
            "user content",
            tokenizer=tokenizer,
        )
        documents = [
            {"doc_id": 12, "text": "doc 12 text"},
            {"doc_id": 49, "text": "doc 49 text"},
        ]
        text = prompt_template.invoke(
            input={
                "documents": documents,
            }
        ).to_string()
        (
            assert_that(text)
            .matches(r"(?ms)<\|start_of_role\|>user<\|end_of_role\|>\s*?user content\s*?<\|end_of_text\|>")
            .contains(*(document["text"] for document in documents))
            .ends_with("<|start_of_role|>assistant<|end_of_role|>")
        )

    @pytest.mark.asyncio
    async def test_documents_async(self, tokenizer):
        assert_that(tokenizer).is_not_none()
        prompt_template = TokenizerChatPromptTemplate.from_template(
            "user content",
            tokenizer=tokenizer,
        )
        documents = [
            {"doc_id": 12, "text": "doc 12 text"},
            {"doc_id": 49, "text": "doc 49 text"},
        ]
        text = (
            await prompt_template.ainvoke(
                input={
                    "documents": documents,
                }
            )
        ).to_string()
        (
            assert_that(text)
            .matches(r"(?ms)<\|start_of_role\|>user<\|end_of_role\|>\s*?user content\s*?<\|end_of_text\|>")
            .contains(*(document["text"] for document in documents))
            .ends_with("<|start_of_role|>assistant<|end_of_role|>")
        )


def identity_llm(prompt: LanguageModelInput) -> str:
    """Mock llm which returns the prompt string as its output"""
    if isinstance(prompt, PromptValue):
        return prompt.to_string()
    if isinstance(prompt, str):
        return prompt
    raise ValueError


class TestDocumentsChain:
    @pytest.mark.parametrize("document_variable_name", ["context", "custom_name"])
    def test_documents_chain(self, tokenizer, document_variable_name):
        assert_that(tokenizer).is_not_none()
        llm = RunnableLambda(identity_llm)
        prompt_template = TokenizerChatPromptTemplate.from_template(
            "user content",
            tokenizer=tokenizer,
        )
        chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template, document_variable_name=document_variable_name)
        documents = [
            Document(page_content="doc 49 text", metadata={"doc_id": 49}),
            Document(page_content="doc 12 text", metadata={"doc_id": 12}),
        ]
        text = chain.invoke(input={document_variable_name: documents})
        (
            assert_that(text)
            .matches(r"(?ms)<\|start_of_role\|>user<\|end_of_role\|>\s*?user content\s*?<\|end_of_text\|>")
            .contains(*(document.page_content for document in documents))
            .ends_with("<|start_of_role|>assistant<|end_of_role|>")
        )

    def test_documents_chain_bind(self, tokenizer):
        assert_that(tokenizer).is_not_none()
        llm = RunnableLambda(identity_llm)
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
        chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)
        documents = [
            Document(page_content="doc 49 text", metadata={"doc_id": 49}),
            Document(page_content="doc 12 text", metadata={"doc_id": 12}),
        ]
        text = chain.invoke(input={"context": documents})
        (
            assert_that(text)
            .matches(r"(?ms)<\|start_of_role\|>user<\|end_of_role\|>\s*?user content\s*?<\|end_of_text\|>")
            .contains(*(document.page_content for document in documents))
            .ends_with("<|start_of_role|>assistant<|end_of_role|>")
        )
