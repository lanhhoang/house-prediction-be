from flask import jsonify
from flask_restful import Resource

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class HousingPrice(Resource):
  def __init__(self):
    self.X = None
    self.Y = None
    self.extracted_X = None
    self.transformed_X = None
    self.x_train = None
    self.x_test = None
    self.y_train = None
    self.y_test = None
    self.r_square = None
    self.y_pred = None

  def get(self):
    self.fetch_data()
    self.transform_data()
    self.split_data()
    self.fit_model()
    return {"r_square": self.r_square, "y_test": self.y_test.tolist(), "y_pred": self.y_pred.tolist()}, 200

  def fetch_data(self):
    self.X, self.Y = fetch_california_housing(return_X_y=True, as_frame=True)
    self.extracted_X = self.X[["AveRooms", "AveBedrms"]]
  
  def transform_data(self):
    transformer = PolynomialFeatures(degree=2, include_bias=False)
    self.transformed_X = transformer.fit_transform(self.extracted_X)
  
  def split_data(self):
    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.transformed_X, self.Y, test_size=10, shuffle=False)
  
  def fit_model(self):
    model = LinearRegression()
    model.fit(self.x_train, self.y_train)
    self.r_square = model.score(self.x_train, self.y_train)
    self.y_pred = model.predict(self.x_test)
