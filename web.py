from flask import Flask
from flask import render_template
app = Flask(__name__)
import pickle
from flask import request

file = open('model.pkl', 'rb')
clf = pickle.load(file)
file.close()

@app.route('/', methods=["GET","POST"])
def hello_world():
    if request.method == "POST":
        mydict = request.form
        print(mydict)
        fever = int(mydict['fever'])
        age = int(mydict['age'])
        pain = int(mydict['pain'])
        Rnose = int(mydict['Rnose'])
        diffbreath = int(mydict['diffbreath'])
                                                  # print(clf.predict([[100, 1, 72, 1, 1]]))
        params = [fever, pain, age, Rnose, diffbreath]
        infprob = clf.predict_proba([params])[0][1]
        print(f"{infprob * 100} %")
        return render_template('show.html', inf = round(infprob*100))
    return render_template("index.html")
 #   return 'Hello, World!'+ str(infprob)

if __name__=="__main__":
    app.run(debug=True)