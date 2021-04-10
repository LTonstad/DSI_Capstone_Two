from flask import Flask, render_template, url_for, request
import markdown
import flask
import markdown.extensions.fenced_code
from pygments.formatters import HtmlFormatter
import numpy as np
from sklearn.datasets import load_iris
import pickle
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import base64
import random
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model

app = Flask(__name__)

# we need to redefine our metric function in order 
# to use it when loading the model 
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    keras.backend.get_session().run(tf.local_variables_initializer())
    return auc

# load the model, and pass in the custom metric function
global graph
graph = tf.get_default_graph()
woj_net = load_model('logcosh.h5', custom_objects={'auc': auc})

@app.route('/')

@app.route('/home', methods=['GET'])
def home():
    return render_template("home.html")

@app.route('/about')
def about():
    readme_file = open("README.md", "r")
    md_template_string = markdown.markdown(
        readme_file.read(), extensions=["fenced_code", "codehilite"]
    )
    
    # Generate Css for syntax highlighting
    formatter = HtmlFormatter(style="murphy",full=True,cssclass="codehilite")
    css_string = formatter.get_style_defs()
    md_css_string = "<style>" + css_string + "</style>"
    
    md_template = md_css_string + md_template_string
    return md_template

@app.route('/EDA')
def EDA():
    column_options = [1, 2, 3, 4]
    return render_template("eda.html", column_options=column_options)

# define a predict function as an endpoint 
@app.route("/predict", methods=["GET","POST"])
def predict():
    data = {"success": False}

    params = flask.request.json
    if (params == None):
        params = flask.request.args

    # if parameters are found, return a prediction
    if (params != None):
        x=pd.DataFrame.from_dict(params, orient='index').transpose()
        with graph.as_default():
            data["prediction"] = str(woj_net.predict(x)[0][0])
            data["success"] = True

    # return a response in json format 
    return flask.jsonify(data) 

@app.route('/results', methods=['POST'])
def results():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    test_data = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)

    filename = 'iris_log_regr.pkl'
    with open(filename, 'rb') as f:
        model = pickle.load(f)

    prediction = model.predict(test_data)

    data = load_iris()
    target_names = data.target_names
    for name in target_names[prediction]:
        predicted_name = name
        
    return render_template('results.html', prediction=predicted_name, image=f'./static/images/iris.png')

if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0') # Should not be in when in production