from flask import Blueprint, render_template, request
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from jinja2 import Template
import pandas as pd


model_name = "deepset/electra-base-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

views = Blueprint('views', __name__)

@views.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        context = request.form.get('context')
        questions = request.form.get('question')
        question_list = []
        question_list = [item for item in str(questions).split(',')]
        answers = []

        for question in question_list:
            QA_input = {
                'question': question,
                'context': context
            }
            res = nlp(QA_input)
            answers.append(res)

        for answer in answers:
            if answer['score'] < 0.01:
                answer.update(answer = "No Answer")
    
        predictions = [list(answer.values())[3] for answer in answers]
        leng = len(question_list)

        return render_template("results.html", context=context, leng=leng, predictions=predictions, question_list=question_list)

    else:
        return render_template("home.html")
