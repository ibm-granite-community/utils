# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import textwrap
from string import Formatter

from dotenv import find_dotenv, load_dotenv


# Function to check if the notebook is running in Google Colab
def is_colab() -> bool:
    try:
        return importlib.util.find_spec("google.colab") is not None
    except ImportError:
        return False


# Function to get the API key
def get_env_var(var_name: str, default_value: str | None = None) -> str:
    """Return the value of the environment variable.

    If the environment variable is not set, search in colab secrets and then .env file.
    If still not found, and default_value is set, the default value is used.
    Otherwise call getpass to ask the user for a value.

    If a value was found in any of the search locations, the value is stored in os.environ.

    Args:
        var_name (str): The environment variable name
        default_value (str | None, optional): A default value to use. Defaults to None.

    Raises:
        ValueError: If a value cannot be located.

    Returns:
        str: The environment variable value.
    """
    # pylint: disable=import-outside-toplevel
    env_var: str | None = os.environ.get(var_name)

    if env_var is not None:
        return env_var

    if is_colab():
        # If in Google Colab, try to get the API key from a secret
        # pylint: disable=import-error
        from google.colab import userdata  # type: ignore[import-not-found]

        try:
            env_var = userdata.get(var_name)
            if env_var:
                print(f"{var_name} loaded from Google Colab secret.")
        except userdata.SecretNotFoundError:
            print(f"{var_name} not found in Google Colab secrets.")

    if not env_var:
        # Try to load API key from .env file
        dotenv_path = find_dotenv(usecwd=True)  # .env can be in a parent folder
        if dotenv_path:
            load_dotenv(dotenv_path=dotenv_path)
            env_var = os.environ.get(var_name)
            if env_var:
                print(f"{var_name} loaded from .env file.")
                return env_var
            print(f"{var_name} not found in .env file.")

    if not env_var and default_value is not None:
        # If we can't find a value in the env, use the default value if provided.
        env_var = default_value

    if not env_var:
        # If neither Colab nor .env file nor default, prompt the user for the API key
        from getpass import getpass

        env_var = getpass(f"Please enter your {var_name}: ")

    if not env_var:
        raise ValueError(f"{var_name} could not be loaded from any source.")

    # Set the environment variable for later implicit access.
    os.environ[var_name] = env_var

    return env_var


def set_env_var(var_name: str, default_value: str | None) -> None:
    """Set the value of the environment variable if it is not set.

    Note: This method has the same behavior as get_env_var.

    Args:
        var_name (str): The environment variable name
        default_value (str | None): A default value to use.
    """
    get_env_var(var_name, default_value)


def wrap_text(text: str, width: int = 80, indent: str = "") -> str:
    """Wrap the specified text to display better in the notebook output.

    Args:
        text (str): The text string to wrap. This string can include multiple lines.
        width (int, optional): The wrapping width. Defaults to 80.
        indent (str, optional): The indent string to use for each line in the result. Defaults to "".

    Returns:
        str: The specified string wrapped to the specified width and indented with the specified indent.
    """
    lines = text.splitlines()
    wrapped_lines = (textwrap.fill(line, width, initial_indent=indent, subsequent_indent=indent) for line in lines)
    return "\n".join(wrapped_lines)


def escape_f_string(f_string: str, *field_names: str) -> str:
    """Escape non-field names in the specified f-string.

    This can be necessary when the f-string contains JSON documents.

    Args:
        f_string (str): The f-string to escape.
        field_names: The field names which are part of the f-string and should not be escaped.

    Returns:
        str: The f-string with non-field names escaped in double braces.
    """
    result = []
    for literal_text, field_name, format_spec, conversion in Formatter().parse(f_string):
        if literal_text:
            result.append(literal_text)
        if field_name is not None:
            is_field = field_name in field_names
            result.append("{" if is_field else "{{")
            result.append(field_name)
            if conversion:
                result.append("!")
                result.append(conversion)
            if format_spec:
                result.append(":")
                result.append(format_spec)
            result.append("}" if is_field else "}}")
    return "".join(result)
