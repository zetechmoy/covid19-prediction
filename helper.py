import numpy as np
import pandas as pd

from keras.models import Model
from keras.utils import plot_model
from keras.layers import Dense, LSTM, Concatenate, Input, Dropout, GRU
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from keras_radam import RAdam
import keras.backend as K

from keras.regularizers import l1, l2, l1_l2

def get_data(csv_path):
	"""
	return preprocessed data
	"""
	#data = np.genfromtxt(csv_path, delimiter=',')
	df = pd.read_csv(csv_path, sep=',')
	df = df.replace(np.nan, '', regex=True)

	############################################################################
	#start by removing states and countries from data
	#since we gonna use lat/long

	df = df.drop(["Province/State", "Country/Region"], axis=1)

	return df

def process_data_values(scaler, data_df, data_vec_size):
	#data values are vecs of size data_vec_size, context are [lat, lng]

	context_x = list()
	values_x = list()

	#add one to data_vec_size because last one will be the y
	data_vec_size = data_vec_size + 1

	data = np.asarray(data_df)
	for i in range(0, data.shape[0]):
		row = data[i]

		latlng = row[0:2]
		values = row[2:-1]

		for j in range(0, values.shape[0]-data_vec_size):
			context_x.append(latlng)
			v = np.asarray([values[j:j+data_vec_size]])
			scaler.partial_fit(v)
			values_x.append(v)

	#for i in range(0, len(values_x)):
	#	values_x[i] = scaler.transform(values_x[i])

	values_x = np.asarray(values_x)

	#we have too much zeros so we delete a part of them
	idx_of_zeros = np.asarray([i for i in range(0, values_x.shape[0]) if (values_x[i] == [0.0]*data_vec_size).all()])
	added_zeros = 0
	max_zeros = int((values_x.shape[0]-idx_of_zeros.shape[0])*0.25)

	values_x = np.delete(values_x, idx_of_zeros[max_zeros:], axis=0)
	context_x = np.delete(context_x, idx_of_zeros[max_zeros:], axis=0)

	return context_x, values_x

def get_location_latlng_switcher(csv_path):
	"""
	return a dict which allow to switch from location to dataset lat/lng
	"""
	df = pd.read_csv(csv_path, sep=',')
	df = df.replace(np.nan, '', regex=True)

	states = df["Province/State"].tolist()
	countries = df["Country/Region"].tolist()

	location_latlng_switch = dict()
	for i in range(0, len(states)):
		id = countries[i] if states[i] == '' else states[i]+"/"+countries[i]
		location_latlng_switch[id] = {"lat": df["Lat"][i], "long": df["Long"][i]}

	return location_latlng_switch

def mish(x):
	return x*K.tanh(K.softplus(x))

def get_model(context_vec_size, values_vec_size):

	context_input = Input(shape=(context_vec_size,))
	values_input = Input(shape=(1, values_vec_size))

	#feature , return_sequences=True
	#values = LSTM(80, activation='relu')(values_input)
	values = GRU(1024, activation="relu", return_sequences=True)(values_input)
	values = Dropout(0.1)(values)
	values = GRU(1024, activation="relu")(values)
	values = Dropout(0.1)(values)

	#merge
	merge = concatenate([values, context_input])

	#output
	#output = Dense(256, activation=mish)(values)
	#output = Dense(128, activation=mish)(output)
	#output = Dense(64, activation=mish)(output)
	#output = Dense(32, activation=mish)(output)
	output = Dense(16, activation="relu")(merge)
	#output = Dense(8, activation=mish)(output)
	output = Dense(8, activation="relu")(output)
	#output = Dense(2, activation=mish)(output)
	output = Dense(1, activation="relu")(output)

	model = Model(inputs=[values_input, context_input], outputs=output)

	opt = Adam(lr=0.001, epsilon=1e-08, decay=0.1)
	model.compile(optimizer=opt, loss='mean_absolute_error', metrics=['acc'])

	# summarize layers
	print(model.summary())
	# plot graph
	plot_model(model, to_file='model.png')

	return model

def get_model_2(context_vec_size, values_vec_size):
	values_input = Input(shape=(1, values_vec_size))

	#feature , return_sequences=True
	#values = LSTM(80, activation='relu')(values_input)
	values = GRU(1024, activation=mish, return_sequences=True)(values_input)
	values = Dropout(0.1)(values)
	values = GRU(1024, activation=mish, return_sequences=True)(values)
	values = Dropout(0.1)(values)
	values = GRU(1024, activation=mish, return_sequences=True)(values)
	values = Dropout(0.1)(values)
	values = GRU(1024, activation=mish)(values)
	values = Dropout(0.1)(values)

	#output
	#output = Dense(256, activation=mish)(values)
	#output = Dense(128, activation=mish)(output)
	#output = Dense(64, activation=mish)(output)
	#output = Dense(32, activation=mish)(output)
	output = Dense(16, activation=mish)(values)
	#output = Dense(8, activation=mish)(output)
	output = Dense(4, activation=mish)(output)
	#output = Dense(2, activation=mish)(output)
	output = Dense(1, activation=mish)(output)

	model = Model(inputs=values_input, outputs=output)

	opt = Adam(lr=0.001, epsilon=1e-08, decay=0.1)
	model.compile(optimizer=opt, loss='mse')

	# summarize layers
	print(model.summary())
	# plot graph
	plot_model(model, to_file='model.png')

	return model
