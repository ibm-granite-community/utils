# SPDX-License-Identifier: Apache-2.0

"""
LangChain support for creating prompt strings using the Transformers
tokenizer's apply_chat_template method.

This is useful for completion models where you need client-side
prompt formatting so the fully-formatter prompt can be sent to the
model.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Annotated, Any, Self

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompt_values import ChatPromptValue, PromptValue
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.prompts.chat import MessageLikeRepresentation
from langchain_core.prompts.string import PromptTemplateFormat
from langchain_core.runnables import RunnableConfig
from pydantic import Field
from typing_extensions import override


def _conversation_message(message: BaseMessage) -> Mapping[str, Any]:
    """Map a messages to a conversation element for apply_chat_template

    Args:
        message (BaseMessage): A formatted BaseMessage.

    Raises:
        ValueError: If the BaseMessage subtype is not understood.

    Returns:
        Mapping[str, Any]: The conversation element.
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
                    "type": "function",
                    "id": tool_call.get("id"),
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
    conversation_message["content"] = message.content
    return conversation_message


class TokenizerChatPromptValue(ChatPromptValue):
    """Tokenizer chat prompt value

    A type of a prompt value whose string is built from messages using
    a transformers tokenizer's apply_chat_template method to format the messages
    into a prompt string.
    """

    apply_chat_template: Annotated[Callable[[Sequence[Mapping[str, Any]]], str], Field(repr=False)]
    """Prompt formatter using transformers tokenizer's apply_chat_template method."""

    @override
    def to_string(self) -> str:
        """Return prompt formatted by a transformers tokenizer's apply_chat_template method."""
        conversation = [_conversation_message(message) for message in self.messages]
        return self.apply_chat_template(conversation)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """This class is not serializable."""
        return False


class TokenizerChatPromptTemplate(ChatPromptTemplate):
    """Tokenizer chat prompt template

    Prompt Template using a transformers tokenizer's apply_chat_template method
    to format the messages into a prompt string.

    Any arguments bound to the prompt will be included in the arguments
    to the apply_chat_template call. For example, you can call bind with
    a tools argument.
    """

    tokenizer: Annotated[Any, Field(repr=False)]
    """The transformers tokenizer for the model to use apply_chat_template
    method to format the prompt string"""

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
            tokenizer: The transformers tokenizer whose apply_chat_template
            method will be used to format the prompt string.
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
    def is_lc_serializable(cls) -> bool:
        """This class is not serializable."""
        return False

    @classmethod
    def from_template(  # type: ignore[override] # pylint: disable=arguments-differ
        cls, template: str, *, tokenizer: Any, **kwargs: Any
    ) -> Self:
        """Create a chat prompt template from a template string.

        Creates a chat template consisting of a single message assumed to be from
        the human.

        Args:
            template: template string
            tokenizer: The transformers tokenizer whose apply_chat_template
            method will be used to format the prompt string.
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
            tokenizer: The transformers tokenizer whose apply_chat_template
                method will be used to format the prompt string.

        Returns:
            A new instance of this class.
        """
        return cls(messages=messages, tokenizer=tokenizer, template_format=template_format)

    def _prompt_value(self, messages: Sequence[BaseMessage], **kwargs: Any) -> TokenizerChatPromptValue:
        """Create a TokenizerChatPromptValue for the formatted messages and kwargs.

        Args:
            messages (list[BaseMessage]): The formatted messages.
            kwargs: Additional arguments to the apply_chat_template method.

        Returns:
            TokenizerChatPromptValue: The PromptValue
        """
        apply_chat_template: Callable[[Sequence[Mapping[str, Any]]], str] = partial(
            self.tokenizer.apply_chat_template,
            tokenize=False,  # output is str
            add_generation_prompt=True,
            **kwargs,
        )
        return TokenizerChatPromptValue(apply_chat_template=apply_chat_template, messages=messages)

    @override
    def format_prompt(self, **kwargs: Any) -> PromptValue:  # type: ignore[override]
        """Format prompt.
            Should return a PromptValue.

        Args:
            **kwargs: Keyword arguments to use for formatting.
                These arguments are used when formatting the message templates
                and are also passed to apply_chat_template.

        Returns:
            PromptValue.
        """
        messages = self.format_messages(**kwargs)
        return self._prompt_value(messages, **kwargs)

    @override
    async def aformat_prompt(self, **kwargs: Any) -> PromptValue:  # type: ignore[override]
        """Async format prompt.
            Should return a PromptValue.

        Args:
            **kwargs: Keyword arguments to use for formatting.
                These arguments are used when formatting the message templates
                and are also passed to apply_chat_template.

        Returns:
            PromptValue.
        """
        messages = await self.aformat_messages(**kwargs)
        return self._prompt_value(messages, **kwargs)

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
