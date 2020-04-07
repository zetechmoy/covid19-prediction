
from helper import *
import numpy as np
import random, os
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler

#define which data we are working on : confirmed, deaths or recovered
data_type = "confirmed"
data_path = "data/time_series_covid19_"+data_type+"_global.csv"

#data vec size represent from how many days we are trying to predict the next day
data_vec_size = 3
context_vec_size = 0 #defined automatically later

training_part = 0.1

def main():

	############################################################################
	#get data and define training and testing data
	data = get_data(data_path)
	scaler = MinMaxScaler(feature_range=(0, 1000))

	#process data
	context_x, values = process_data_values(scaler, data, data_vec_size)

	print("context_x", len(context_x))
	print("values", len(values))
	context_vec_size = len(context_x[0])

	#shuffle and cut data
	shuf = list(zip(context_x, values))
	random.shuffle(shuf)
	context_x, values = zip(*shuf)
	context_x, values = np.asarray(context_x), np.asarray(values)

	#extract x and y from values
	values_x = values[:,:,0:-1]
	values_y = values[:,:,-1].reshape((1, -1))[0]

	cut_index = int(values_x.shape[0]*training_part)

	print("context_x", context_x.shape, context_x[0])
	print("values_x", values_x.shape, values_x[0])
	print("values_y", values_y.shape, values_y[0])

	#now data are ready !

	############################################################################
	#Define the model
	model = get_model(context_vec_size, data_vec_size)

	#pretty print on tensorboard (, callbacks=[tensorboard_callback])
	#logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
	#tensorboard_callback = TensorBoard(log_dir=logdir)

	#model.fit([context_x, values_x], values_y, batch_size=128, epochs=50)
	for _ in range(0, 10):
		model.fit([values_x, context_x], values_y, batch_size=256, epochs=5, validation_split=training_part)
		test_sample_context_x = context_x[0:3]
		test_sample_x = values_x[0:3]
		test_sample_y = values_y[0:3]
		predictions = model.predict([test_sample_x, context_x])

		#inverse transform
		predictions_transformed = scaler.inverse_transform(np.hstack((predictions, np.zeros((predictions.shape[0], data_vec_size)))))[:,0]
		test_sample_y_transformed = scaler.inverse_transform(np.hstack((test_sample_y.reshape((-1, 1)), np.zeros((predictions.shape[0], data_vec_size)))))[:,0]

		for i in range(0, test_sample_x.shape[0]):
			#data without inverse_transform
			print(test_sample_context_x[i].tolist(), test_sample_x[i].tolist()[0], "=>", predictions[i].tolist()[0], "/", test_sample_y[i])

			#inverse_transformed data
			#test_sample_x_transformed = scaler.inverse_transform(np.hstack((test_sample_x[i], [[0]])))[0][0:-1]
			#print(test_sample_context_x[i].tolist(), test_sample_x_transformed, "=>", predictions_transformed[i], "/", test_sample_y_transformed[i])

	modeldir = "models/" + datetime.now().strftime("%Y%m%d-%H%M%S")
	# Save the model
	os.mkdir(modeldir)
	model.save(modeldir+"/"+data_type+"_model.h5")

if __name__ == '__main__':
	main()
