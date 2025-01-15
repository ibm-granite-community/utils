# Utils tests

import getpass
import pathlib

from assertpy import assert_that

from ibm_granite_community.notebook_utils import get_env_var, set_env_var


# Test retrieval of new environment variable
def test_env_var_set(monkeypatch):
    monkeypatch.setenv("TEST_VAR", "bartholomew")

    assert_that(get_env_var("TEST_VAR")).is_equal_to("bartholomew")


# Test retrieval of existing environment variable
def test_env_var_preset():
    assert_that(get_env_var("PATH")).is_not_none()


# Test acquisition of environment variable using getpass
def test_env_var_getpass(monkeypatch):
    monkeypatch.setattr(getpass, "getpass", lambda prompt: "abc123")

    assert_that(get_env_var("APIKEY")).is_equal_to("abc123")


# Test retrieval of environment variable from .env file,
# for cases where there is a .env file, and assuming dotenv works
def test_env_var_dotenv():
    env_path = pathlib.Path(".env")

    assert_that(str(env_path)).does_not_exist()
    try:
        env_path.write_text("TEST_API_KEY=xyz123", encoding="utf-8")
        assert_that(get_env_var("TEST_API_KEY")).is_equal_to("xyz123")
    finally:
        env_path.unlink()


# Test fallback to default environment variable
def test_env_var_default():
    assert_that(get_env_var("FAVORITE_COLOR", "blue")).is_equal_to("blue")
    assert_that(get_env_var("FAVORITE_COLOR")).is_equal_to("blue")


def test_set_env_var_default():
    set_env_var("FAVORITE_COLOR2", "green")
    assert_that(get_env_var("FAVORITE_COLOR2")).is_equal_to("green")
