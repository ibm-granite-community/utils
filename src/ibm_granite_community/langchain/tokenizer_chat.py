# SPDX-License-Identifier: Apache-2.0

"""LangChain support for creating prompt strings using the Transformers
tokenizer's apply_chat_template method.

Also includes a create_stuff_documents_chain method for building a RAG chain
which works with a transformers tokenizer's apply_chat_template method's documents argument.
"""

import json
from collections.abc import Sequence
from typing import Any, cast

from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompt_values import ChatPromptValue, PromptValue
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.prompts.chat import MessageLikeRepresentation
from langchain_core.prompts.string import PromptTemplateFormat
from langchain_core.runnables import Runnable, RunnableConfig, RunnablePassthrough
from typing_extensions import Self, override


def _conversation_message(message: BaseMessage) -> dict[str, Any]:
    """Map a messages to a conversation element for apply_chat_template

    Args:
        message (BaseMessage): A formatted BaseMessage.

    Raises:
        ValueError: If the BaseMessage subtype is not understood.

    Returns:
        dict[str, Any]: The conversation element.
    """
    conversation_message: dict[str, Any] = {
        key: getattr(message, key)
        for key in message.model_fields_set
        if key not in BaseMessage.model_fields  # pylint: disable=unsupported-membership-test
    }

    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
        if message.tool_calls:  # Fix up for OpenAI conventions
            conversation_message["tool_calls"] = [
                {
                    "id": tool_call.get("id"),
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": json.dumps(tool_call["args"]),
                    },
                }
                for tool_call in message.tool_calls
            ]
    elif isinstance(message, SystemMessage):
        role = "system"
    elif isinstance(message, ToolMessage):
        role = "tool"
    elif isinstance(message, ChatMessage):
        role = message.role.lower()
    else:
        msg = f"Got unsupported message type: {message}"
        raise ValueError(msg)  # noqa: TRY004

    conversation_message["role"] = role
    conversation_message["content"] = (
        message.text()
        if not isinstance(message.content, str) and all(isinstance(item, str) or (item.get("type") == "text" and isinstance(item.get("text"), str)) for item in message.content)
        else message.content
    )
    return conversation_message


class TokenizerChatPromptValue(ChatPromptValue):
    """Tokenizer chat prompt value

    A type of a prompt value that is built from messages using
    a transformers tokenizer apply_chat_template method to format the messages
    into a prompt string.
    """

    text: str
    """Prompt formatted by a transformers tokenizer apply_chat_template method."""

    @override
    def to_string(self) -> str:
        """Return prompt formatted by a transformers tokenizer apply_chat_template method."""
        return self.text

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "prompts", "tokenizer"]


class TokenizerChatPromptTemplate(ChatPromptTemplate):
    """Tokenizer chat prompt template

    Prompt Template using a transformers tokenizer apply_chat_template method
    to format the messages into a prompt string.

    Any arguments bound to the prompt will be included in the arguments
    to the apply_chat_template call. For example, you can call bind with
    a tools argument.
    """

    tokenizer: Any
    """The transformers tokenizer for the model to use apply_chat_template
    method to format the prompt"""

    def __init__(
        self,
        messages: Sequence[MessageLikeRepresentation],
        *,
        tokenizer: Any,
        template_format: PromptTemplateFormat = "f-string",
        **kwargs: Any,
    ) -> None:
        """Create a TokenizerChatPromptTemplate.

        Args:
            messages (Sequence[MessageLikeRepresentation]): The messages for the prompt.
            tokenizer: The transformers tokenizer to use apply_chat_template
            method to format the prompt.
            template_format (PromptTemplateFormat, optional): The format for the message
            templates. Defaults to "f-string".
        """
        super().__init__(
            messages=messages,
            template_format=template_format,
            tokenizer=tokenizer,
            **kwargs,
        )

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "prompts", "tokenizer"]

    @classmethod
    def from_template(  # type: ignore[override] # pylint: disable=arguments-differ
        cls, template: str, *, tokenizer: Any, **kwargs: Any
    ) -> Self:
        """Create a chat prompt template from a template string.

        Creates a chat template consisting of a single message assumed to be from
        the human.

        Args:
            template: template string
            tokenizer: The transformers tokenizer to use apply_chat_template
            method to format the prompt.
            **kwargs: keyword arguments to pass to the constructor.

        Returns:
            A new instance of this class.
        """
        prompt_template = PromptTemplate.from_template(template, **kwargs)
        message = HumanMessagePromptTemplate(prompt=prompt_template)
        return cls.from_messages(messages=[message], tokenizer=tokenizer)

    @classmethod
    def from_messages(  # type: ignore[override] # pylint: disable=arguments-differ
        cls,
        messages: Sequence[MessageLikeRepresentation],
        template_format: PromptTemplateFormat = "f-string",
        *,
        tokenizer: Any,
    ) -> Self:
        """Create a chat prompt template from a variety of message formats.

        Examples:
            Instantiation from a list of message templates:

            .. code-block:: python

                template = TokenizerChatPromptTemplate.from_messages([
                    ("human", "Hello, how are you?"),
                    ("ai", "I'm doing well, thanks!"),
                    ("human", "That's good to hear."),
                ])

            Instantiation from mixed message formats:

            .. code-block:: python

                template = TokenizerChatPromptTemplate.from_messages([
                    SystemMessage(content="hello"),
                    ("human", "Hello, how are you?"),
                ])

        Args:
            messages: sequence of message representations.
                  A message can be represented using the following formats:
                  (1) BaseMessagePromptTemplate, (2) BaseMessage, (3) 2-tuple of
                  (message type, template); e.g., ("human", "{user_input}"),
                  (4) 2-tuple of (message class, template), (5) a string which is
                  shorthand for ("human", template); e.g., "{user_input}".
            template_format: format of the message templates. Defaults to "f-string".
            tokenizer: The transformers tokenizer to use apply_chat_template
                method to format the prompt.

        Returns:
            A new instance of this class.
        """
        return cls(messages=messages, tokenizer=tokenizer, template_format=template_format)

    def _apply_chat_template(self, messages: list[BaseMessage], **kwargs: Any) -> TokenizerChatPromptValue:
        """Apply the tokenizer's chat template to the formatted messages and kwargs.

        Args:
            messages (list[BaseMessage]): The formatted messages.
            kwargs: Additional arguments to the apply_chat_template method.

        Returns:
            TokenizerChatPromptValue: The PromptValue
        """
        conversation = [_conversation_message(message) for message in messages]
        prompt = cast(
            str,
            self.tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,  # output is str
                add_generation_prompt=True,
                **kwargs,
            ),
        )
        return TokenizerChatPromptValue(text=prompt, messages=messages)

    @override
    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Format prompt using tokenizer apply_chat_template.
            Should return a PromptValue.

        Args:
            **kwargs: Keyword arguments to use for formatting.
                These arguments are used when formatting the message templates
                and are also passed to apply_chat_template.

        Returns:
            PromptValue.
        """
        messages = self.format_messages(**kwargs)
        return self._apply_chat_template(messages, **kwargs)

    @override
    async def aformat_prompt(self, **kwargs: Any) -> PromptValue:
        """Async format prompt using tokenizer apply_chat_template.
            Should return a PromptValue.

        Args:
            **kwargs: Keyword arguments to use for formatting.
                These arguments are used when formatting the message templates
                and are also passed to apply_chat_template.

        Returns:
            PromptValue.
        """
        messages = await self.aformat_messages(**kwargs)
        return self._apply_chat_template(messages, **kwargs)

    @override
    def invoke(
        self,
        input: dict,  # pylint: disable=redefined-builtin
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> PromptValue:
        """Invoke the prompt.

        Args:
            input: Dict, input to the prompt.
            config: RunnableConfig, configuration for the prompt.
            kwargs: Additional input arguments. This will include
                arguments bound to this prompt.

        Returns:
            PromptValue: The output of the prompt.
        """
        input_ = kwargs | input
        return super().invoke(input=input_, config=config, **kwargs)

    @override
    async def ainvoke(
        self,
        input: dict,  # pylint: disable=redefined-builtin
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> PromptValue:
        """Async invoke the prompt.

        Args:
            input: Dict, input to the prompt.
            config: RunnableConfig, configuration for the prompt.
            kwargs: Additional input arguments. This will include
                arguments bound to this prompt.

        Returns:
            PromptValue: The output of the prompt.
        """
        input_ = kwargs | input
        return await super().ainvoke(input=input_, config=config, **kwargs)


PromptTemplateLike = BasePromptTemplate | Runnable[dict[str, Any], PromptValue]


def create_stuff_documents_chain(
    llm: LanguageModelLike,
    prompt: PromptTemplateLike,
    *,
    output_parser: BaseOutputParser | None = None,
    document_variable_name: str = "context",
) -> Runnable[dict[str, Any], Any]:
    """Create a chain for passing a list of Documents to a model
        using a TokenizerChatPromptTemplate.

    Args:
        llm: Language model.
        prompt: Tokenizer chat prompt template. Prepared documents will be
            passed in using the input variable "documents".
        output_parser: Output parser. Defaults to StrOutputParser.
        document_variable_name: Variable name to use for the input documents to be prepared.
            Defaults to "context" which is the name used by create_retrieval_chain.

    Returns:
        An LCEL Runnable. The input is a dictionary that must have a "context" key (override by
            setting document_variable_name) that
            maps to a List[Document], and any other input variables expected in the prompt.
            The Runnable return type depends on output_parser used.
    """

    _output_parser = output_parser or StrOutputParser()

    def prepare_documents(inputs: dict[str, Any]) -> list[dict[str, str]]:
        documents: list[Document] = inputs[document_variable_name]
        return [{**document.metadata, "text": document.page_content} for document in documents]

    return (RunnablePassthrough.assign(documents=prepare_documents).with_config(run_name="prepare_documents") | prompt | llm | _output_parser).with_config(
        run_name="stuff_documents_chain"
    )
