from flask import Flask
from flask_restful import Api, Resource
from housing_price import HousingPrice

app = Flask(__name__)
api = Api(app)

api.add_resource(HousingPrice, "/predict")

if __name__ == "__main__":
  app.run(debug=True)
