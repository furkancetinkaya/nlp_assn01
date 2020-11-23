#!/usr/bin/python

## Step_1: Tokenize
## Step_2: Remove punctuation
## Step_3: Lemmatization

import os
import nltk
import string
nltk.download('punkt',download_dir="./env/nltk_data/")
nltk.download('wordnet',download_dir="./env/nltk_data/")

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from keras.preprocessing.text import text_to_word_sequence
from lemminflect import getLemma

##=================================================================================================##
## Library
def read_text_source():
    with open("text_source.txt", "r") as fdesc:
        source_txt = fdesc.read()
    return source_txt

def remove_punctuation(source):
    return [c for c in source if c not in string.punctuation]

def lemmatize_with_wordnet(source):
    lemmer = WordNetLemmatizer()
    lemmed_txt = []
    for w in source:
        lemmed_txt.append(lemmer.lemmatize(w, pos="v"))
    return lemmed_txt

def lemmatize_with_lemminflect(source):
    lemmed = []
    for w in source:
        lemmed.append(getLemma(w,upos='VERB'))
    return lemmed

##=================================================================================================##
## Methods
def method1(srcText):
    tokenized = word_tokenize(srcText, language="english")
    cleaned_txt = remove_punctuation(tokenized)
    lemmatized = lemmatize_with_wordnet(cleaned_txt)
    print("NLTK Tokenizer, WordNet Lemmatizer\n----------------------------------")
    print(tokenized, end="\n\n")
    print(lemmatized, end="\n\n")
    return True

def method2(srcText):
    tokenized = text_to_word_sequence(srcText)
    lemmatized = lemmatize_with_wordnet(tokenized)
    print("Keras Tokenizer, WordNet Lemmatizer\n----------------------------------")
    print(tokenized, end="\n\n")
    print(lemmatized, end="\n\n")
    return True

def method3(srcText):
    tokenized = word_tokenize(srcText, language="english")
    cleaned_txt = remove_punctuation(tokenized)
    lemmatized = lemmatize_with_lemminflect(cleaned_txt)
    print("NLTK Tokenizer, LemmInflect Lemmatizer\n----------------------------------")
    print(tokenized, end="\n\n")
    print(lemmatized, end="\n\n")
    return True

def method4(srcText):
    tokenized = text_to_word_sequence(srcText)
    lemmatized = lemmatize_with_lemminflect(tokenized)
    print("Keras Tokenizer, LemmInflect Lemmatizer\n----------------------------------")
    print(tokenized, end="\n\n")
    print(lemmatized, end="\n\n")
    return True

##=================================================================================================##
## MAIN CODE

os.system('cls' if os.name == 'nt' else 'clear')        # Clear the console

source_text = read_text_source()                        # Read the sample text
print("Original Text\n----------------------\n", source_text, end="\n\n")

method1_res = method1(source_text)      # Execute methods
method2_res = method2(source_text)
method3_res = method3(source_text)
method4_res = method4(source_text)

