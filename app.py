import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

#Loading the model
logisticmodel = pickle.load(open("logistic_multinomial.pkl","rb"))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    data_values = np.array(list(data.values()))
    print(data_values)
    output = logisticmodel.predict(data_values)
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug = True)


