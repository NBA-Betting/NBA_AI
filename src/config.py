"""
config.py

This module is responsible for loading and managing the application's configuration settings. 
It utilizes environment variables for dynamic configuration and reads settings from a YAML file. 

The module performs the following main tasks:
1. Loads environment variables from a .env file to ensure they are available within the application.
2. Loads configuration data from 'config.yaml', a YAML configuration file.
3. Replaces placeholders in the configuration values with corresponding environment variables, 
   allowing for dynamic and flexible configuration management.
4. Computes absolute file paths for specific configuration settings, such as database paths, 
   ensuring consistent and correct file access regardless of the working directory.
5. Ensures a secret key is set for the web application, generating one if necessary.

Main Functions:
- `load_config()`: Loads and processes the application configuration.

Global Variables:
- `config`: A dictionary containing the processed configuration settings, accessible globally 
  within the application.
"""

import os

import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def load_config():
    """
    Loads the application configuration from a YAML file and environment variables.

    This function reads the configuration from 'config.yaml', replaces placeholders
    in the configuration values with the corresponding environment variables (if any),
    and computes absolute paths where necessary. It supports dynamic configuration
    by allowing environment variables to override settings in the YAML file.

    Returns:
        dict: A dictionary containing the application configuration.
    """
    # Open and read the YAML configuration file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Function to replace placeholders in the config values with environment variables
    def replace_env_vars(value):
        """
        Replaces placeholders in the given value with environment variables.

        If the value is a string that starts with '${' and ends with '}', it is
        considered a placeholder for an environment variable. The function attempts
        to replace it with the value of the environment variable. If the environment
        variable is not set, the original value (including the placeholder) is returned.

        Args:
            value (str): The configuration value that may contain a placeholder.

        Returns:
            str: The value with the placeholder replaced by the environment variable's value,
                 or the original value if the environment variable is not set.
        """
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]  # Extract the environment variable name
            return os.getenv(env_var, value)  # Replace with env var value, if available
        return value

    # Iterate through the config dictionary to replace placeholders with env vars
    for section, settings in config.items():
        for key, value in settings.items():
            config[section][key] = replace_env_vars(value)

    # Compute the absolute path for DB_PATH based on PROJECT_ROOT
    # This is useful for ensuring that file paths are correctly handled regardless of the working directory
    project_root = config["project"]["root"]
    if "database" in config and "path" in config["database"]:
        config["database"]["path"] = os.path.join(
            project_root, config["database"]["path"]
        )

    # Ensure the secret key is set
    if config["web_app"]["secret_key"] in ["${WEB_APP_SECRET_KEY}", ""]:
        config["web_app"]["secret_key"] = os.urandom(24).hex()  # Generate a random key

    return config


# Load the configuration into a global variable for easy access throughout the application
config = load_config()
