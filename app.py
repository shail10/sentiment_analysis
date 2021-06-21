from flask import Flask, render_template, redirect, url_for, flash, session
from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from wtforms.validators import DataRequired
import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
import re

model = load_model('sent_model.h5')
word_to_id = imdb.get_word_index()

def pred_on_new_data(model,sample_json,word_to_id=word_to_id):
    max_len = 2697
    string = sample_json['text']
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    
    string = string.lower().replace("<br />", " ")
    
    string=re.sub(strip_special_chars, "", string)
    
    words = string.split()
    
    test = [[word_to_id[word] if word in word_to_id else 0 for word in words]]
    
    test = pad_sequences(test,maxlen = max_len)
    
    pred_prob = model.predict(test)
    pred_class = model.predict_classes(test)
    
    return (pred_prob[0][0],pred_class[0][0])

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'


class SentForm(FlaskForm):
	text = TextAreaField('Enter the text',validators=[DataRequired()])
	submit = SubmitField('Analyze')



@app.route('/',methods=['GET','POST'])
def index():

	form = SentForm()
	if form.validate_on_submit():
		session['text'] = form.text.data
		return redirect(url_for("prediction"))
	return render_template('home.html',form=form)



@app.route('/new')
def new():
	form  = SentForm()
	return render_template('layout.html',form=form)



@app.route('/prediction')
def prediction():
	content = {}
	sentiment = ''
	content['text'] = str(session['text'])
	pred_prob,pred_class = pred_on_new_data(model=model,sample_json=content)
	if pred_class == 1:
		sentiment = 'POSITIVE'
	else:
		sentiment = 'NEGATIVE'
	return render_template('prediction.html',pred_prob=pred_prob,pred_class=pred_class,sentiment=sentiment)
 


if __name__ == '__main__':
    app.run(debug=True)