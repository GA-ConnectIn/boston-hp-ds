# This script runs the application on a local server.
# It contains the definition of routes and views for the application.

import flask
import numpy as np
import pandas as pd

#---------- MODEL IN MEMORY ----------------#

# Read in the titanic data and build a model on it
from sklearn.datasets import load_boston
boston = load_boston()
df = pd.DataFrame(boston.data, columns = boston.feature_names)

# Create dummies and drop NaN
y = boston.target
X = df[["CRIM", "AGE", "DIS"]]

from sklearn.linear_model import LinearRegression

lr = LinearRegression(normalize = True)

PREDICTOR = lr.fit(X, y)





#---------- CREATING AN API, METHOD 1 ----------------#

# Initialize the app
app = flask.Flask(__name__)


# When you navigate to the page 'server/predict', this will run
# the predict() function on the parameters in the url.
#
# Example URL:

#http://10.242.248.169:4000/predict?crim=2&age=2&dis=2
@app.route('/predict', methods=["GET"])
def predict():
    '''Makes a prediction'''
    crim = float(flask.request.args['crim'])
    age = float(flask.request.args['age'])
    dis = float(flask.request.args['dis'])

    item = np.array([crim, age, dis]).reshape(1,-1)
    score = PREDICTOR.predict(item)[0]
    results = {'housePrice': score, 'houseList': list(y)}
    return flask.jsonify(results)



if __name__ == '__main__':
    '''Connects to the server'''

    HOST = "127.0.0.1"
    PORT = 4000

    app.run(HOST, PORT)
