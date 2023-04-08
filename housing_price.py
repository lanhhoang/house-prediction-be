from flask import jsonify, request
from flask_restful import Resource

import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class HousingPrice(Resource):
  def __init__(self):
    self.r_square = None
    self.y_pred = None
  
  def get(self):
    return {"message": "Send a POST request to make a prediction"}, 200

  def post(self):
    data = request.get_json()
    aveRooms = float(data.get("aveRooms"))
    aveBedrms = float(data.get("aveBedrms"))
    self.train_model()
    self.predict(aveRooms, aveBedrms)
    return {"r_square": self.r_square, "y_pred": self.y_pred.tolist()}, 200
  
  def predict(self, aveRooms, aveBedrms):
    model, transformer = self.load_model_transformer()
    input_data = np.array([aveRooms, aveBedrms]).reshape(1, -1)
    transformed_input_data = transformer.fit_transform(input_data)
    self.y_pred = model.predict(transformed_input_data)
  
  def train_model(self):
    # Fetch dataset
    feature, target = fetch_california_housing(return_X_y=True, as_frame=True)
    extracted_feature = feature[["AveRooms", "AveBedrms"]]

    # Transform data using Polynomial Regression
    transformer = PolynomialFeatures(degree=2, include_bias=False)
    transformed_extracted_feature = transformer.fit_transform(extracted_feature)

    # Split input data to train data and test data
    x_train, x_test, y_train, y_test = train_test_split(transformed_extracted_feature, target, test_size=0.2, shuffle=False)

    # Fit model
    model = LinearRegression()
    model.fit(x_train, y_train)
    self.r_square = model.score(x_train, y_train)

    # Save the trained model and transformer to files
    joblib.dump(model, "trained_model.pkl")
    joblib.dump(transformer, "trained_transformer.pkl")

  def load_model_transformer(self):
    return [joblib.load("trained_model.pkl"), joblib.load("trained_transformer.pkl")]
