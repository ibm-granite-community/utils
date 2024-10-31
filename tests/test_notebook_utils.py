# Utils tests

import getpass
import os

import dotenv

from ibm_granite_community.notebook_utils import get_env_var, set_env_var


# Test retrieval of new environment variable
def test_env_var_set():
    os.environ["TEST_VAR"] = "bartholomew"
    assert get_env_var("TEST_VAR") == "bartholomew"


# Test retrieval of existing environment variable
def test_env_var_preset():
    assert get_env_var("REPLICATE_API_TOKEN") is not None


# Test acquisition of environment variable using getpass
def test_env_var_getpass(monkeypatch):
    monkeypatch.setattr(getpass, "getpass", lambda prompt: "abc123")
    assert get_env_var("APIKEY") == "abc123"


# Test retrieval of environment variable from .env file,
# for cases where there is a .env file, and assuming dotenv works
def test_env_var_dotenv(monkeypatch):
    assert not os.path.exists(".env")
    monkeypatch.setattr(os.path, "exists", lambda x: True)

    def set_api_key():
        os.environ["TEST_API_KEY"] = "xyz123"

    monkeypatch.setattr(dotenv, "load_dotenv", set_api_key)
    assert get_env_var("TEST_API_KEY") == "xyz123"


# Test fallback to default environment variable
def test_env_var_default():
    assert get_env_var("FAVORITE_COLOR", "blue") == "blue"


def test_set_env_var_default():
    set_env_var("FAVORITE_COLOR", "blue")
    assert get_env_var("FAVORITE_COLOR") == "blue"
