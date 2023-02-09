from flask import Flask, request, app, jsonify, url_for, render_template
import pickle
import pandas as pd
import numpy as np
app =Flask(__name__)
# load model
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods =['POST'])
def predict():
    data = request.json['data']
    print("data received:", data)
    print(np.array(list(data.values())).reshape(1,-1))
    pickled_model = pickle.load(open('regression_model.pkl','rb'))
    pickled_std = pickle.load(open('std.pkl', 'rb'))    
    scaled_data = pickled_std.transform(np.array(list(data.values())).reshape(1,-1))
    predicted = pickled_model.predict(scaled_data)
    print("predictted :", predicted)
    return jsonify(predicted[0])

if __name__ == "__main__":
    app.run(debug = True)
# new_sample1 = pickled_std.transform(boston.data[0].reshape(1,-1))
# pickled_model.predict(new_sample1)