# SPDX-License-Identifier: Apache-2.0

"""
LangChain support utils methods.
"""

from collections import deque

from langchain_core.language_models import LanguageModelLike
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.runnables import RunnableBinding, RunnableLambda, RunnableSequence


def is_chat_model(llm: LanguageModelLike, default: bool = False) -> bool:
    """
    This method will attempt to discern if the input is a chat model
    (extends BaseChatModel class) or a completion model (extends BaseLLM class).
    If a determination cannot be made, it returns the default value.
    """
    queue = deque[LanguageModelLike]()
    queue.append(llm)
    while queue:
        candidate = queue.pop()
        match candidate:
            case BaseLLM():
                return False
            case BaseChatModel():
                return True
            case RunnableBinding():
                queue.append(candidate.bound)
            case RunnableLambda():
                queue.extend(candidate.deps)
            case RunnableSequence():
                queue.extend(candidate.steps)
            case _:
                pass
    return default
