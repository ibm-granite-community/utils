import os

# Function to check if the notebook is running in Google Colab
def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

# Function to get the API key
def get_api_key():
    api_key = None

    if is_colab():
        # If in Google Colab, try to get the API key from a secret
        from google.colab import userdata
        try:
            api_key = userdata.get('REPLICATE_API_TOKEN')
            if api_key:
                print("API key loaded from Google Colab secret.")
        except userdata.SecretNotFoundError:
            print("REPLICATE_API_TOKEN not found in Google Colab secrets.")

    if not api_key and os.path.exists('.env'):
        # Try to load API key from .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ModuleNotFoundError:
            print("Module 'dotenv' not found. Please install it using 'pip install python-dotenv'.")
        api_key = os.getenv('REPLICATE_API_TOKEN')
        if api_key:
            print("API key loaded from .env file.")
        else:
            print("REPLICATE_API_TOKEN not found in .env file.")
    if not api_key:
        # If neither Colab nor .env file, prompt the user for the API key
        from getpass import getpass
        api_key = getpass("Please enter your API key: ")

    if not api_key:
        raise ValueError("API key could not be loaded from any source.")

    return api_key
