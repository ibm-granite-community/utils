# SPDX-License-Identifier: Apache-2.0

"""
LangChain support for a create_stuff_documents_chain method for building a RAG chain
which uses passes a 'documents' argument to the prompt and the model.

This is important for chat models where you need to pass the `documents` argument in the request.
It is also useful for completion models when using TokenizerChatPromptTemplate to format the
prompt with the model's tokenizer.
"""

from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelInput, LanguageModelLike, LanguageModelOutput
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, chain

PromptTemplateLike = BasePromptTemplate | Runnable[dict[str, Any], PromptValue]


def create_stuff_documents_chain(
    llm: LanguageModelLike,
    prompt: PromptTemplateLike,
    *,
    output_parser: BaseOutputParser | None = None,
    document_variable_name: str = "context",
) -> Runnable[dict[str, Any], Any]:
    """Create a chain for passing a list of Documents to a prompt and a model
        so each can use documents.

    Args:
        llm: Language model. Prepared documents will be
            passed in using the keyword argument "documents".
        prompt: Prompt template. Prepared documents will be
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

    @chain
    def prepare_documents(inputs: dict[str, Any]) -> list[dict[str, str]]:
        documents: list[Document] = inputs[document_variable_name]
        return [{**document.metadata, "text": document.page_content} for document in documents]

    @chain
    def invoke_llm(inputs: dict[str, Any]) -> LanguageModelOutput:
        prompt_value: LanguageModelInput = inputs["prompt_value"]
        documents: list[dict[str, str]] = inputs["documents"]
        return llm.invoke(prompt_value, documents=documents)

    return (
        RunnablePassthrough.assign(documents=prepare_documents).with_config(run_name="prepare_documents")
        | RunnablePassthrough.assign(prompt_value=prompt).with_config(run_name="format_prompt")
        | invoke_llm
        | _output_parser
    ).with_config(run_name="stuff_documents_chain")
