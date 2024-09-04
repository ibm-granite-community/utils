# my_granite_package/setup.py

from setuptools import setup, find_packages

setup(
    name='granite_community_utils',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'python-dotenv'  # Ensure this is installed if .env loading is needed
    ],
    author='The IBM Granite Community Team',
    author_email='trevor.d.grant@gmail.com',
    description='A utility package of utility functions for IBM Granite Community notebooka.',
    url='https://github.com/ibm-granite-community/utils',  # Replace with your actual repo URL
    keywords='notebook colab api key granite',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
