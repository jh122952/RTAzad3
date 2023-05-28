import pickle
from flask import Flask
from flask import request
from flask import jsonify
import numpy as np

# Create a flask
app = Flask(__name__)

class FirstPerceptron():
    def __init__(self, n_features):
         self.n_f = n_features

    def predict(self, x):
        z = np.dot(x, self.w) + self.b
        pred = np.where(z > 0, 1, 0)
        return pred

    def fit(self, X, y, n_epochs):
         
        self.n_epochs = n_epochs 
        self.w = np.zeros((self.n_f,), dtype = float)
        self.b = float(0)

        for i in range(self.n_epochs):
            for xi, yi in zip(X, y):
                pred_y = self.predict(xi)
                error = yi - pred_y
                self.w = self.w + error * xi
                self.b += error
        return self

# Create an API end point
@app.route('/api/predict', methods=['GET'])
def get_prediction():

    sepal_length = float(request.args.get('sl'))
    petal_length = float(request.args.get('pl'))
    
    features = [sepal_length,
                petal_length]
    
    print(features)

    # Load pickled model file
    with open('model.pkl',"rb") as picklefile:
        model = pickle.load(picklefile)
    print(model)
    # Predict the class using the model
    predicted_class = int(model.predict(features))
    
    # Return a json object containing the features and prediction
    return jsonify(features=features, predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)


