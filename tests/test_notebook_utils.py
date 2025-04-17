# Utils tests

import getpass
import pathlib

import pytest
from assertpy import assert_that

from ibm_granite_community.notebook_utils import escape_f_string, get_env_var, set_env_var, wrap_text


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


@pytest.mark.parametrize("width,indent", [(30, ""), (50, ""), (80, ""), (30, "  "), (50, "* "), (80, "> "), (200, ""), (200, "> ")])
def test_wrap_text(width: int, indent: str):
    wide_string = """\
This string is long and thus should be wrapped so that there are multiple lines each less than the requested width.
"""
    wrapped = wrap_text(wide_string, width=width, indent=indent)
    splitlines = wrapped.splitlines()
    if len(wide_string) > width:
        assert_that(len(splitlines), "len(splitlines)").is_greater_than_or_equal_to(2)
    else:
        assert_that(len(splitlines), "len(splitlines)").is_equal_to(1)
    for line in splitlines:
        assert_that(len(line), "len(line)").is_less_than_or_equal_to(width)
        if len(indent) > 0:
            assert_that(line).starts_with(indent)


@pytest.mark.parametrize(
    "f_string,expected,field_names",
    [
        ("foo", "foo", []),
        ("foo {bar}", "foo {bar}", ["bar"]),
        ("foo {bar}", "foo {{bar}}", []),
        ("foo {bar} {baz} fum", "foo {bar} {{baz}} fum", ["bar"]),
        ("foo {bar} {baz} fum", "foo {bar} {baz} fum", ["baz", "bar"]),
    ],
)
def test_escape_f_string(f_string: str, expected: str, field_names: list[str]):
    assert_that(escape_f_string(f_string, *field_names)).is_equal_to(expected)
