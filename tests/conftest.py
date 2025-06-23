# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase


@pytest.fixture(scope="session")
def tokenizer() -> PreTrainedTokenizerBase:
    model_path = "ibm-granite/granite-3.3-8b-instruct"
    return AutoTokenizer.from_pretrained(model_path)
