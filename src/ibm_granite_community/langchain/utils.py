# SPDX-License-Identifier: Apache-2.0

"""
LangChain support utils methods.
"""

from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any, cast

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel, LanguageModelLike
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.messages import BaseMessage, ChatMessage
from langchain_core.runnables import RunnableBinding, RunnableLambda, RunnableSequence


def find_model(candidate: LanguageModelLike) -> BaseLanguageModel | None:
    """
    This method will attempt to discern the base language model.
    If a determination cannot be made, it returns None.
    """
    queue = deque[LanguageModelLike]()
    queue.append(candidate)
    while queue:
        candidate = queue.pop()
        match candidate:
            case BaseLanguageModel():
                return candidate
            case RunnableBinding():
                queue.append(candidate.bound)
            case RunnableLambda():
                queue.extend(candidate.deps)
            case RunnableSequence():
                queue.extend(candidate.steps)
            case _:
                pass
    return None


def is_chat_model(llm: BaseLanguageModel | None, default: bool = False) -> bool:
    """
    This method will attempt to discern if the input is a chat model
    (extends BaseChatModel class) or a completion model (extends BaseLLM class).
    If a determination cannot be made, it returns the default value.
    """
    match llm:
        case BaseLLM():
            return False
        case BaseChatModel():
            return True
        case _:
            return default


def add_document_role_messages(messages: Sequence[BaseMessage], documents: Sequence[Document] | Sequence[Mapping[str, Any]]) -> list[BaseMessage]:
    """Add document role messages for the specified documents.

    Args:
        messages (Sequence[BaseMessage]): The initial messages
        documents (Sequence[Document] | Sequence[Mapping[str, Any]]): The documents.

    Returns:
        list[BaseMessage]: A list including the initial messages and document role
        messages for each of the specified documents.
    """
    if not documents:  # no documents
        return list(messages)

    document_messages: list[BaseMessage]
    if isinstance(documents[0], Document):  # list[Document]
        document_messages = [
            ChatMessage(role=f"document {document.metadata.get('doc_id', i)}", content=document.page_content)
            for i, document in enumerate(cast(Sequence[Document], documents), start=1)
        ]
    else:  # list[Mapping[str, Any]]
        document_messages = [
            ChatMessage(role=f"document {document.get('doc_id', i)}", content=document["text"])  #
            for i, document in enumerate(cast(Sequence[Mapping[str, Any]], documents), start=1)
        ]

    document_messages.extend(messages)
    return document_messages
