from flask import Flask,render_template,request
app = Flask(__name__)
import pickle
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

file = open('model.pkl','rb')
clf = pickle.load(file)
file.close()

@app.route('/',methods=["GET","POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        fever = int(myDict['fever'])
        age = int(myDict['age'])
        pain = int(myDict['pain'])
        breath = int(myDict['breath'])
        nose = int(myDict['nose'])

    

        inputFeatures =[fever,pain,age,nose,breath]
        infoProb=clf.predict_proba([inputFeatures])[0][1]
        print(infoProb)
        return render_template('show.html',inf=round(infoProb*100))
        # return render_template('show.html',inf=infoProb)
        # return 'the information is'+str(infoProb)
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)
        