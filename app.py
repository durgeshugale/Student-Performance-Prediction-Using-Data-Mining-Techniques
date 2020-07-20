from flask import Flask, render_template, request
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
ml_model = pickle.load(open("model.pkl", "rb"))



#@app.route('/test')
#def test():
#    return 'Flask is being used for Development'


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        try:

            name = (request.form['q1'])
            gender = (request.form['q2'])
            if gender == 'M':
                gender = 0
            else:
                gender = 1
            age = int(request.form['q3'])
            medu = int(request.form['q4'])
            fedu = int(request.form['q5'])
            studytime = int(request.form['q6'])
            failures = 0
            famrel = int(request.form['q8'])
            freetime = int(request.form['q9'])
            health = int(request.form['q10'])
            absences = int(request.form['q11'])

            address = (request.form['q12'])
            if address == 'U':
                address = 0
            else:
                address = 1
            guardian = (request.form['q13'])
            if guardian == 'mother':
                guardian = 0
            else:
                guardian = 1

            pred_args = [gender , age, medu, fedu, studytime, failures, famrel, freetime, health, absences, address]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1, -1)
            print(pred_args_arr.reshape(1, -1))
            #mul_reg = open("multiple_linear_model.pkl", "rb")
            #ml_model = joblib.load(mul_reg)
            model_prediction = ml_model.predict(pred_args_arr)
            print(ml_model.predict(pred_args_arr))
            model_prediction = round(float(model_prediction))
            print(round(float(model_prediction)))
        except ValueError:
            return "Please check if the values are entered correctly"
    return render_template('prediction.html', prediction = model_prediction, name= name)

if __name__ == '__main__':
    app.run()