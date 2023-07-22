from flask import Flask, render_template
import joblib
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import numpy as np
from train_model import tokenizer

model = load_model('my_model.h5')

app = Flask(__name__)

from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired

app.config.update(dict(
    SECRET_KEY="powerful secretkey",
    WTF_CSRF_SECRET_KEY="a csrf secret key"
))

class MyForm(FlaskForm):
    name = StringField('name', validators=[DataRequired()])

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    form = MyForm()
    if form.validate_on_submit():
        rew=[f"{form.name}"]
        rewt = tokenizer.texts_to_sequences(rew)

        a=model.predict(rewt)
        b=round(float(a*10))
        if b<1:
            b=1
        if b<6:
            ret = f'{b},  negative'
            return ret 
        else:
            ret = f'{b},  positive'
            return ret
    
    return render_template('submit.html', form=form)