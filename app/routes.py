from flask import Flask, render_template, request, Response
from flask_wtf import FlaskForm
import json, keras
import numpy as np
from keras import backend as K
import tensorflow as tf
from datetime import date, datetime

from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model


app = Flask(__name__)

def auc(y_true, y_pred):
	auc = tf.metrics.auc(y_true, y_pred)[1]
	keras.backend.get_session().run(tf.local_variables_initializer())
	return auc

def get_diff_day():
	d0 = date(2020, 1, 22)
	d1 = datetime.now().date()
	d = d1-d0
	return d.days

#tf_config = some_custom_config
sess = tf.Session()
graph = tf.get_default_graph()

set_session(sess)
model_confirmed = keras.models.load_model("models/20200407-225637/confirmed_model.h5", custom_objects={'auc': auc})
#model_death = keras.models.load_model("models/20200407-225637/death_model.h5")
#model_recovered = keras.models.load_model("models/20200407-225637/recovered_model.h5")

class ModelLoader(object):

	def __init__(self, file_path):
		self.name = file_path
		self.file_path = file_path
		self.model = keras.models.load_model(self.file_path)
		self.graph = tf.get_default_graph()

	def predict(self, to_predict):
		#model = keras.models.load_model(self.file_path)
		with self.graph.as_default():
			predict = self.model.predict(to_predict)
		#del model
		return predict


#model = ModelLoader("models/20200407-225637/confirmed_model.h5")

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
	global graph

	datatype = request.form.get("datatype", None)
	last1 = request.form.get("last1", None)
	last2 = request.form.get("last2", None)
	last3 = request.form.get("last3", None)
	lat = request.form.get("lat", None)
	lng = request.form.get("lng", None)

	latlng = np.asarray([lat, lng], dtype=np.float32)
	input = np.asarray([last1, last2, last3], dtype=np.float32)
	duration = get_diff_day()

	global sess
	global graph
	with graph.as_default():
		set_session(sess)
		input = [[[input]], [latlng], [duration]]
		prediction = round(model_confirmed.predict(input).tolist()[0][0])

	return Response(json.dumps({"prediction": prediction}))
