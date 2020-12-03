#!/usr/bin/bash

# Firstly create a virtual environment
python -m venv env

# Then install dependeicies
pip install nltk     # Natural Language Processing Toolkit
pip install trnlp    # Turkish NLP Library
pip install zeyrek   # Another NLP Library for Turkish

# Then download required packages
python reqs.py

# Print success message
echo "If there is no error, the environment setup is done :)"