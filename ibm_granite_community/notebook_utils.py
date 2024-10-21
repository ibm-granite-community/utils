import os

# Function to check if the notebook is running in Google Colab
def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

# Function to get the API key
def get_env_var(var_name, default_value=None):
    env_var = None

    if os.environ.get(var_name) is not None:
        return os.environ.get(var_name)

    if is_colab():
        # If in Google Colab, try to get the API key from a secret
        from google.colab import userdata
        try:
            env_var = userdata.get(var_name)
            if env_var:
                print(f"{var_name} loaded from Google Colab secret.")
        except userdata.SecretNotFoundError:
            print(f"{var_name} not found in Google Colab secrets.")

    if not env_var and os.path.exists('.env'):
        # Try to load API key from .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ModuleNotFoundError:
            print("Module 'dotenv' not found. Please install it using 'pip install python-dotenv'.")
        env_var = os.getenv(var_name)
        if env_var:
            print(f"{var_name} loaded from .env file.")
        else:
            print(f"{var_name} not found in .env file.")

    if not env_var:
        # If we can't find a value in the env, use the default value if provided.
        if default_value is not None:
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

def set_env_var(var_name, default_value=None):
    if os.environ.get(var_name) is None:
        os.environ[var_name] = get_env_var(var_name, default_value=default_value)
