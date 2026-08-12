# IBM Granite Community Utils

A Python package providing utility functions for IBM Granite Community notebooks and recipes, designed to simplify working with LLMs, LangChain, and various notebook environments.

## Build Status

[![CI Build](https://github.com/ibm-granite-community/utils/actions/workflows/unit-testing.yaml/badge.svg)](https://github.com/ibm-granite-community/utils/actions/workflows/unit-testing.yaml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-Apache-yellow.svg)](https://opensource.org/licenses/APACHE-2.0)

## Features

- **Environment Detection**: Automatically detect and adapt to different notebook environments (Google Colab, local Jupyter, etc.)
- **Secure Configuration Management**: Load API keys and environment variables from multiple sources (environment, .env files, Google Colab secrets, or user prompts)
- **LangChain Utilities**: Enhanced utilities for working with LangChain, including:
  - Model detection and type checking
  - Document passing via `chat_template_kwargs`, document role messages, or tool call messages
  - Custom prompt templates with tokenizer support
  - Advanced document chain implementations
- **Text Processing**: Utilities for text wrapping and f-string escaping
- **Type Safety**: Full type hints and py.typed support

## Installation

### Basic Installation

```bash
uv pip install ibm-granite-community-utils
```

### With Optional Dependencies

For tokenizer chat prompt support:

```bash
uv pip install "ibm-granite-community-utils[tokenizer_chat]"
```

### Development Installation

```bash
git clone https://github.com/ibm-granite-community/utils.git
cd utils
uv sync
```

## Requirements

- Python 3.11 or higher
- Core dependencies:
  - `python-dotenv>=1.0.0`
  - `langchain-core>=1.0.0`
  - `typing-extensions>=4.0.0`

## Usage

### Environment Variable Management

```python
from ibm_granite_community.notebook_utils import get_env_var, set_env_var

# Get an API key from environment, .env file, Colab secrets, or prompt user
api_key = get_env_var("WATSONX_API_KEY")

# Set an environment variable with a default value
set_env_var("MODEL_ID", "ibm-granite/granite-4.1-8b")
```

### Environment Detection

```python
from ibm_granite_community.notebook_utils import is_colab

if is_colab():
    print("Running in Google Colab")
else:
    print("Running in local environment")
```

### Text Processing

```python
from ibm_granite_community.notebook_utils import wrap_text, escape_f_string

# Wrap long text for better display
wrapped = wrap_text("Your long text here...", width=80, indent="  ")

# Escape f-strings containing JSON or other brace-heavy content
escaped = escape_f_string('{"field": {value}}', "value")
```

### LangChain Utilities

```python
from ibm_granite_community.langchain.utils import (
    find_model,
    is_chat_model,
    add_document_role_messages,
    add_tool_call_messages,
)

# Find the base language model in a chain
model = find_model(your_chain)

# Check if a model is a chat model
if is_chat_model(model):
    print("This is a chat model")

# Add document role messages (for APIs that don't support chat_template_kwargs
# but whose model chat template understands the document role, e.g. older Granite models on Ollama)
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

messages = [HumanMessage(content="What is in the documents?")]
documents = [Document(page_content="Document content here")]
enhanced_messages = add_document_role_messages(messages, documents)

# Add tool call messages (documents serialized as XML, for models with tool-calling support)
enhanced_messages = add_tool_call_messages(messages, documents)
```

### Tokenizer Chat Prompts

```python
from ibm_granite_community.langchain.prompts import TokenizerChatPromptTemplate
from transformers import AutoTokenizer

# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-4.1-8b")

# Create a chat prompt template that uses the tokenizer for formatting
prompt = TokenizerChatPromptTemplate.from_messages(
    messages=[
        ("system", "You are a helpful assistant."),
        ("user", "{question}")
    ],
    tokenizer=tokenizer
)

# Use the prompt
formatted = prompt.format_prompt(question="What is AI?")
print(formatted.to_string())
```

### Custom Document Chains

```python
from ibm_granite_community.langchain.chains.combine_documents import (
    DocumentMode,
    create_stuff_documents_chain,
)

# Default: pass documents via chat_template_kwargs (works with most hosted APIs)
chain = create_stuff_documents_chain(llm, prompt)

# Use document role messages when the API doesn't support chat_template_kwargs
# but the model's chat template understands the document role (e.g. older Granite models on Ollama)
chain = create_stuff_documents_chain(llm, prompt, document_mode=DocumentMode.DOCUMENT_ROLES)

# Use tool call messages when the API doesn't support documents parameter in chat_template_kwargs
# but the model supports tool calling
chain = create_stuff_documents_chain(llm, prompt, document_mode=DocumentMode.TOOL_CALL)
```

## API Reference

### `notebook_utils` Module

- **`is_colab() -> bool`**: Check if running in Google Colab
- **`get_env_var(var_name: str, default_value: str | None = None) -> str`**: Get environment variable from multiple sources
- **`set_env_var(var_name: str, default_value: str | None) -> None`**: Set environment variable if not already set
- **`wrap_text(text: str, width: int = 80, indent: str = "") -> str`**: Wrap text for display
- **`escape_f_string(f_string: str, *field_names: str) -> str`**: Escape non-field names in f-strings

### `langchain.utils` Module

- **`find_model(candidate: LanguageModelLike) -> BaseLanguageModel | None`**: Find base language model in a chain
- **`is_chat_model(llm: BaseLanguageModel | None, default: bool = False) -> bool`**: Check if model is a chat model
- **`add_document_role_messages(messages, documents) -> list[BaseMessage]`**: Prepend document role `ChatMessage` objects for each document
- **`add_tool_call_messages(messages, documents) -> list[BaseMessage]`**: Append an AI tool call message and a `ToolMessage` with documents serialized as XML

### `langchain.chains.combine_documents` Module

- **`DocumentMode`**: Enum controlling how documents are passed to the model — `CHAT_TEMPLATE_KWARGS`, `DOCUMENT_ROLES`, or `TOOL_CALL` (default)
- **`create_stuff_documents_chain(llm, prompt, *, output_parser, document_variable_name, document_mode) -> Runnable`**: Create a chain that stuffs retrieved documents into the prompt and model invocation

## Contributing

For information about contributing to this repo, code of conduct guidelines, etc., see the community [CONTRIBUTING][CG] and [Code of Conduct][CoC] guides. All commits require [DCO-signoff][CG-legal] _and_ [GPG or SSH signing][CG-signing]. The GitHub recommended code security settings are enforced on this public repository (which include the signing requirement).

For more background, please see the [community discussions](https://github.com/orgs/ibm-granite-community/discussions).

## Licenses

Code in this repository is licensed under Apache 2.0.

## IBM Public Repository Disclosure

All content in these repositories including code has been provided by IBM under the associated open source software license and IBM is under no obligation to provide enhancements, updates, or support. IBM developers produced this code as an open source project (not as an IBM product), and IBM makes no assertions as to the level of quality nor security, and will not be maintaining this code going forward.

[CoC]: https://github.com/ibm-granite-community/.github/blob/main/CODE_OF_CONDUCT.md
[CG]: https://github.com/ibm-granite-community/.github/blob/main/CONTRIBUTING.md
[CG-legal]: https://github.com/ibm-granite-community/.github/blob/main/CONTRIBUTING.md#legal
[CG-signing]: https://github.com/ibm-granite-community/.github/blob/main/CONTRIBUTING.md#signing-commits
