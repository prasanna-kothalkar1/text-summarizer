from flask import Flask, render_template, request
import numpy as np
import os
from model import summ1,summ2
#import seaborn as sns
#import matplotlib.pyplot as plt
#import pandas as pd

app = Flask(__name__)


UPLOAD_FOLDER = '/Users/adityavs14/Documents/Internship/Pianalytix/Month_2/ASL/app/static'
ALLOWED_EXTENSIONS = set(['png','jpg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER





@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        file1 = request.form['file1']
        s1 = summ1(file1)
        s2 = summ2(file1)
    return render_template('index.html',inp=file1,result1 = s1, result2=s2) 





if __name__ == "__main__":
    app.run(debug=True)