# Import required packages
import pandas as pd
import numpy as np
import spacy
from flask import Flask, render_template, request
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

@app.route('/')
def initialize_page():
    global trained
    global questions_answers
    global question_vec
    # Read trained word set and FAQ corpus
    trained = spacy.load('en_core_web_md')
    questions_answers = pd.read_csv('Mental_Health_FAQ.csv')

    # Drop unnecessary column
    questions_answers.drop(['Question_ID'],axis=1)

    # Vectorize Questions
    question_vec = []
    for x in questions_answers['Questions'].values:
        question_vec.append(trained(x).vector)
    question_vec = np.stack(question_vec, axis=0)
    return render_template("index.html")


@app.route('/', methods=['POST'])
def answer_question():
    global trained
    global question_vec
    global questions_answers
    # Get user question
    question = request.form["questionArea"]
    # Append user question to other questions
    svd = TruncatedSVD(n_components=10)
    my_sentence_vec = np.stack([trained(question).vector])
    new_question_vec = question_vec
    new_question_vec = np.append(new_question_vec, my_sentence_vec, axis=0)
    vector_size = len(new_question_vec)
    # Compute similarity
    svd_sentences = svd.fit_transform(new_question_vec)
    cos_sim = cosine_similarity(svd_sentences, svd_sentences)
    # Get the most similar one
    objects = []
    counter = 0
    for elem in cos_sim[vector_size-1]:
        obj = {}
        obj['index'] = counter
        obj['value'] = elem
        objects.append(obj)
        counter += 1
    idx = sorted(objects, key=lambda x: x['value'], reverse=True)[1]['index']
    sim_ques = questions_answers['Questions'][idx]
    sim_answ = questions_answers['Answers'][idx]
    return render_template("answer.html", sim_ques=sim_ques, answer=sim_answ)


if __name__ == '__main__':
   app.run()