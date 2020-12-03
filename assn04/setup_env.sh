#!/usr/bin/bash

# Firstly create a virtual environment
python -m venv env

# Then install dependencies
pip install -r requirements.txt

# Then download required packages
python reqs.py

# Print success message
echo "If there is no error, the environment setup is done :)"