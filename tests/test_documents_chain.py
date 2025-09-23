# SPDX-License-Identifier: Apache-2.0

# create_stuff_documents_chain tests

import json
from typing import Any

import pytest
from assertpy import assert_that
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, convert_to_openai_messages
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.outputs import ChatResult, LLMResult
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from transformers import PreTrainedTokenizerBase

from ibm_granite_community.langchain.chains.combine_documents import create_stuff_documents_chain
from ibm_granite_community.langchain.prompts import TokenizerChatPromptTemplate
from ibm_granite_community.langchain.utils import add_document_role_messages, is_chat_model


# Method to use as a tool
def i_am_a_tool(tool_arg: str) -> str:
    """I am a tool!

    Args:
        tool_arg: The tool argument

    Returns:
        str: The tool argument.
    """
    return tool_arg


class MockLLM(BaseLLM):
    """Mock llm which returns the formatted prompt, messages and kwargs in its output"""

    def invoke(  # pylint: disable=redefined-builtin
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        prompt_value = self._convert_input(input)
        # Use client-side prompt formatting
        prompt = prompt_value.to_string()
        result = json.dumps(dict(kwargs, prompt=prompt, messages=[message.model_dump(exclude_none=True) for message in prompt_value.to_messages()]))
        return result

    async def ainvoke(  # pylint: disable=redefined-builtin
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        return self.invoke(input, config, stop=stop, **kwargs)

    def _generate(
        self,
        prompts: list[str],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        raise NotImplementedError()

    @property
    def _llm_type(self) -> str:
        return "test"


class MockChat(BaseChatModel):
    """Mock chat llm which returns the formatted prompt, messages and kwargs in its output"""

    tokenizer: PreTrainedTokenizerBase

    def invoke(  # pylint: disable=redefined-builtin
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> BaseMessage:
        prompt_value = self._convert_input(input)
        # Emulate server-side prompt formatting (don't call prompt_value.to_string())
        conversation = convert_to_openai_messages(prompt_value.to_messages())
        if not isinstance(conversation, list):
            conversation = [conversation]
        prompt = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,  # output is str
            add_generation_prompt=True,
            **kwargs,
        )
        result = json.dumps(dict(kwargs, prompt=prompt, messages=[message.model_dump(exclude_none=True) for message in prompt_value.to_messages()]))
        return AIMessage(result)

    async def ainvoke(  # pylint: disable=redefined-builtin
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> BaseMessage:
        return self.invoke(input, config, stop=stop, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError()

    @property
    def _llm_type(self) -> str:
        return "test"

    def bind_tools(
        self,
        tools: list[dict[str, Any]],  # type: ignore[override]
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        return self.bind(tools=tools)


class TestDocumentsChain:
    @pytest.mark.parametrize("document_variable_name", ["context", "custom_name"])
    def test_documents_chain(self, tokenizer, documents: list[Document], document_variable_name):
        assert_that(tokenizer).is_not_none()
        llm = MockLLM()
        prompt_template = TokenizerChatPromptTemplate.from_template(
            "user content",
            tokenizer=tokenizer,
        )
        chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template, document_variable_name=document_variable_name, output_parser=JsonOutputParser())
        result = chain.invoke(input={document_variable_name: documents})
        assert_that(result).contains("prompt")
        (
            assert_that(result["prompt"])
            .matches(r"(?ms)<\|start_of_role\|>user<\|end_of_role\|>\s*?user content\s*?<\|end_of_text\|>")
            .contains(*(document.page_content for document in documents))
            .ends_with("<|start_of_role|>assistant<|end_of_role|>")
        )
        assert_that(result).contains("messages")
        assert_that(result["messages"]).is_length(1)
        assert_that(result["messages"]).extracting("content", filter={"type": "human"}).contains("user content")
        assert_that(result).does_not_contain("documents")

    @pytest.mark.parametrize("document_variable_name", ["context", "custom_name"])
    def test_documents_chain_chat(self, tokenizer, documents: list[Document], document_variable_name):
        assert_that(tokenizer).is_not_none()
        llm = MockChat(tokenizer=tokenizer)
        prompt_template = ChatPromptTemplate.from_template(
            "user content",
        )
        chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template, document_variable_name=document_variable_name, output_parser=JsonOutputParser())
        result = chain.invoke(input={document_variable_name: documents})
        assert_that(result).contains("prompt")
        (
            assert_that(result["prompt"])
            .matches(r"(?ms)<\|start_of_role\|>user<\|end_of_role\|>\s*?user content\s*?<\|end_of_text\|>")
            .contains(*(document.page_content for document in documents))
            .ends_with("<|start_of_role|>assistant<|end_of_role|>")
        )
        assert_that(result).contains("messages")
        assert_that(result["messages"]).is_length(1)
        assert_that(result["messages"]).extracting("content", filter={"type": "human"}).contains("user content")
        assert_that(result).contains("documents")
        assert_that(result["documents"]).extracting("text").contains(*(document.page_content for document in documents))
        assert_that(result["documents"]).extracting("doc_id").contains(*(document.metadata["doc_id"] for document in documents))

    def test_documents_chain_bind(self, tokenizer, documents: list[Document]):
        assert_that(tokenizer).is_not_none()
        tools = [convert_to_openai_tool(i_am_a_tool)]
        llm = MockLLM().bind(tools=tools)
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
        result = chain.invoke(input={"context": documents})
        assert_that(result).contains("prompt")
        (
            assert_that(result["prompt"])
            .matches(r"(?ms)<\|start_of_role\|>user<\|end_of_role\|>\s*?user content\s*?<\|end_of_text\|>")
            .contains(*(document.page_content for document in documents))
            .ends_with("<|start_of_role|>assistant<|end_of_role|>")
        )
        assert_that(result).contains("messages")
        assert_that(result["messages"]).is_length(1)
        assert_that(result["messages"]).extracting("content", filter={"type": "human"}).contains("user content")
        assert_that(result).contains("tools")
        assert_that(result["tools"]).is_length(1)
        assert_that(result["tools"]).extracting("type").contains_only(tools[0]["type"])
        assert_that(result["tools"]).extracting("function").extracting("name").contains_only(tools[0]["function"]["name"])
        assert_that(result).does_not_contain("documents")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("document_variable_name", ["context", "custom_name"])
    async def test_documents_chain_async(self, tokenizer, documents: list[Document], document_variable_name):
        assert_that(tokenizer).is_not_none()
        llm = MockLLM()
        prompt_template = TokenizerChatPromptTemplate.from_template(
            "user content",
            tokenizer=tokenizer,
        )
        chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template, document_variable_name=document_variable_name, output_parser=JsonOutputParser())
        result = await chain.ainvoke(input={document_variable_name: documents})
        assert_that(result).contains("prompt")
        (
            assert_that(result["prompt"])
            .matches(r"(?ms)<\|start_of_role\|>user<\|end_of_role\|>\s*?user content\s*?<\|end_of_text\|>")
            .contains(*(document.page_content for document in documents))
            .ends_with("<|start_of_role|>assistant<|end_of_role|>")
        )
        assert_that(result).contains("messages")
        assert_that(result["messages"]).is_length(1)
        assert_that(result["messages"]).extracting("content", filter={"type": "human"}).contains("user content")
        assert_that(result).does_not_contain("documents")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("document_variable_name", ["context", "custom_name"])
    async def test_documents_chain_chat_async(self, tokenizer, documents: list[Document], document_variable_name):
        assert_that(tokenizer).is_not_none()
        llm = MockChat(tokenizer=tokenizer)
        prompt_template = ChatPromptTemplate.from_template(
            "user content",
        )
        chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template, document_variable_name=document_variable_name, output_parser=JsonOutputParser())
        result = await chain.ainvoke(input={document_variable_name: documents})
        assert_that(result).contains("prompt")
        (
            assert_that(result["prompt"])
            .matches(r"(?ms)<\|start_of_role\|>user<\|end_of_role\|>\s*?user content\s*?<\|end_of_text\|>")
            .contains(*(document.page_content for document in documents))
            .ends_with("<|start_of_role|>assistant<|end_of_role|>")
        )
        assert_that(result).contains("messages")
        assert_that(result["messages"]).is_length(1)
        assert_that(result["messages"]).extracting("content", filter={"type": "human"}).contains("user content")
        assert_that(result).contains("documents")
        assert_that(result["documents"]).extracting("text").contains(*(document.page_content for document in documents))
        assert_that(result["documents"]).extracting("doc_id").contains(*(document.metadata["doc_id"] for document in documents))

    @pytest.mark.parametrize("use_document_roles", [False, True])
    def test_documents_chain_bind_chat(self, tokenizer, documents: list[Document], use_document_roles):
        assert_that(tokenizer).is_not_none()
        tools = [convert_to_openai_tool(i_am_a_tool)]
        llm = MockChat(tokenizer=tokenizer).bind_tools(tools=tools)
        prompt_template = ChatPromptTemplate.from_messages(
            messages=[
                MessagesPlaceholder("user_content"),
            ],
        )
        chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template, output_parser=JsonOutputParser(), use_document_roles=use_document_roles)
        result = chain.invoke(
            input={
                "context": documents,
                "user_content": [
                    HumanMessage(content="user content"),
                ],
            }
        )
        assert_that(result).contains("prompt")
        (
            assert_that(result["prompt"])
            .matches(r"(?ms)<\|start_of_role\|>user<\|end_of_role\|>\s*?user content\s*?<\|end_of_text\|>")
            .contains(*(document.page_content for document in documents))
            .ends_with("<|start_of_role|>assistant<|end_of_role|>")
        )
        assert_that(result).contains("messages")
        assert_that(result["messages"]).is_length(len(documents) + 1 if use_document_roles else 1)
        assert_that(result["messages"]).extracting("content", filter={"type": "human"}).contains("user content")
        if use_document_roles:
            assert_that(result).does_not_contain("documents")
            assert_that(result["messages"]).extracting("content", filter={"type": "chat"}).contains(*(document.page_content for document in documents))
            assert_that(result["messages"]).extracting("role", filter={"type": "chat"}).contains(*(f"document {document.metadata['doc_id']}" for document in documents))
        else:
            assert_that(result).contains("documents")
            assert_that(result["documents"]).extracting("text").contains(*(document.page_content for document in documents))
            assert_that(result["documents"]).extracting("doc_id").contains(*(document.metadata["doc_id"] for document in documents))
        assert_that(result).contains("tools")
        assert_that(result["tools"]).is_length(1)
        assert_that(result["tools"]).extracting("type").contains_only(tools[0]["type"])
        assert_that(result["tools"]).extracting("function").extracting("name").contains_only(tools[0]["function"]["name"])

    @pytest.mark.parametrize("llm_cls", [MockLLM, MockChat])
    def test_is_chat_model(self, tokenizer, llm_cls: type):
        assert_that(tokenizer).is_not_none()
        llm = llm_cls(tokenizer=tokenizer)
        expected = isinstance(llm, BaseChatModel)
        description = f"{llm_cls} {'is a' if expected else 'is not a'} chat model"
        assert_that(is_chat_model(llm)).described_as(description).is_equal_to(expected)
        bound_llm = llm.bind(foo="bar")
        assert_that(is_chat_model(bound_llm)).described_as(f"Bound {description}").is_equal_to(expected)
        lambda_llm = RunnableLambda(lambda inputs: llm.invoke(inputs))  # pylint: disable=unnecessary-lambda
        assert_that(is_chat_model(lambda_llm)).described_as(f"Lambda {description}").is_equal_to(expected)
        sequence_llm = RunnableLambda(lambda x: x) | bound_llm | JsonOutputParser()
        assert_that(is_chat_model(sequence_llm)).described_as(f"Sequence {description}").is_equal_to(expected)

    def test_add_document_role_messages_lc(self, documents: list[Document]):
        assert_that(documents).is_not_empty()
        prompt_template = ChatPromptTemplate.from_template("user content")
        messages = prompt_template.format_messages()
        document_messages = add_document_role_messages(messages, documents)
        assert_that(document_messages).is_length(len(messages) + len(documents))
        message_dicts = [message.model_dump(exclude_none=True) for message in document_messages]
        assert_that(message_dicts).extracting("content", filter={"type": "human"}).contains("user content")
        assert_that(message_dicts).extracting("content", filter={"type": "chat"}).contains(*(document.page_content for document in documents))
        assert_that(message_dicts).extracting("role", filter={"type": "chat"}).contains(*(f"document {document.metadata['doc_id']}" for document in documents))

    def test_add_document_role_messages_dict(self, documents: list[Document]):
        assert_that(documents).is_not_empty()
        prompt_template = ChatPromptTemplate.from_template("user content")
        messages = prompt_template.format_messages()
        document_dicts: list[dict[str, Any]] = [{**document.metadata, "text": document.page_content} for document in documents]
        document_messages = add_document_role_messages(messages, document_dicts)
        assert_that(document_messages).is_length(len(messages) + len(document_dicts))
        message_dicts = [message.model_dump(exclude_none=True) for message in document_messages]
        assert_that(message_dicts).extracting("content", filter={"type": "human"}).contains("user content")
        assert_that(message_dicts).extracting("content", filter={"type": "chat"}).contains(*(document["text"] for document in document_dicts))
        assert_that(message_dicts).extracting("role", filter={"type": "chat"}).contains(*(f"document {document.metadata['doc_id']}" for document in documents))

    def test_add_document_role_messages_empty(self):
        prompt_template = ChatPromptTemplate.from_template("user content")
        messages = prompt_template.format_messages()
        document_messages = add_document_role_messages(messages, [])
        assert_that(document_messages).is_length(len(messages))
        message_dicts = [message.model_dump(exclude_none=True) for message in document_messages]
        assert_that(message_dicts).extracting("content", filter={"type": "human"}).contains("user content")
        assert_that(message_dicts).extracting("content", filter={"type": "chat"}).is_empty()
