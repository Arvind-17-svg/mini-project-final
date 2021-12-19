import numpy as np
from flask import Flask, request, render_template
import pickle
import os

app=Flask(__name__)
model=pickle.load(open('rdf.pkl','rb'))

@app.route('/')
def main():
    return render_template('home.html')




@app.route('/predict',methods=['POST'])
def home():
    data1 = float(request.form['N'])
    data2 = float(request.form['P'])
    data3 = float(request.form['K'])
    data4 = float(request.form['temperature'])
    data5 = float(request.form['humidity'])
    data6 = float(request.form['ph'])
    data7 = float(request.form['rainfall'])
    
    
    arr = np.array([[data1, data2, data3, data4, data5, data6, data7]])
    print(arr)
    pred = model.predict(arr)
    return render_template('about.html', data=pred)



if __name__ == "__main__":
    app.run(debug=True)
