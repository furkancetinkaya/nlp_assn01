#!/usr/bin/bash

# Firstly create a virtual environment
python -m venv env

# Then install dependencies
pip install -r requirements.txt

# Then download required packages
python reqs.py

# Download the corpus, and convert it to txt
python convert_csv2txt.py
