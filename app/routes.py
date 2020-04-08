from flask import Flask, render_template, request, Response
from flask_wtf import FlaskForm
import json, keras
import numpy as np
from keras import backend as K
import tensorflow as tf
from datetime import date, datetime


app = Flask(__name__)

K.clear_session()


def get_diff_day():
	d0 = date(2020, 1, 22)
	d1 = datetime.now().date()
	d = d1-d0
	return d.days

#model_confirmed = keras.models.load_model("models/20200407-225637/confirmed_model.h5")
#model_death = keras.models.load_model("models/20200407-225637/death_model.h5")
#model_recovered = keras.models.load_model("models/20200407-225637/recovered_model.h5")

class ModelLoader(object):

	def __init__(self, file_path):
		self.name = file_path
		self.file_path = file_path

	def predict(self, to_predict):
		model = keras.models.load_model(self.file_path)
		predict = model.predict(to_predict)
		del model
		return predict

models = {
	"confirmed": "models/20200407-225637/confirmed_model.h5",
	"deaths": "models/20200407-225637/deaths_model.h5",
	"recovered": "models/20200407-225637/recovered_model.h5"
}

@app.route("/")
def map():
	return render_template('form.html')

@app.route("/predict", methods=["POST"])
def predict():
	#datatype

	datatype = request.form.get("datatype", None)
	last1 = request.form.get("last1", None)
	last2 = request.form.get("last2", None)
	last3 = request.form.get("last3", None)
	lat = request.form.get("lat", None)
	lng = request.form.get("lng", None)

	latlng = np.asarray([lat, lng], dtype=np.float32)
	input = np.asarray([last1, last2, last3], dtype=np.float32)
	duration = get_diff_day()

	model = ModelLoader(models[datatype])

	input = [[[input]], [latlng], [duration]]
	prediction = round(model.predict(input).tolist()[0][0])
	return Response(json.dumps({"prediction": prediction}))
