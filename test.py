import keras
import numpy as np

data_type = "confirmed"
model_path = "models/20200406-181926/"+data_type+"_model.h5"

#china [27.3338, 128.5697],[56.0, 90.0, 114.0] => 146.55475
#suisse [46.8182, 8.2275],[56.0, 90.0, 114.0] => 146.9167

all_values = [404, 632, 921]

latlng = np.asarray([46.131357,-2.4364346])

last_evolution = np.asarray(all_values)

for i in range(0, 10):
	print(last_evolution)
	input = [[[last_evolution]], [latlng]]

	# Recreate the exact same model purely from the file
	model = keras.models.load_model(model_path)

	prediction = model.predict(input)[0][0]
	next_last_evolution = np.hstack((last_evolution[1:], [prediction]))
	all_values.append(prediction)
	last_evolution = next_last_evolution

print(all_values)


#check whether location as a real influence (just a bit)
#for i in range(0, 180, 1):
#	for j in range(0, 180, 1):
#		latlng = np.asarray([j, i])
#		last_evolution = np.asarray([56.0, 90.0, 114.0])
#		input = [[[last_evolution]], [latlng]]
#		prediction = model.predict(input)
#
#		print([j, i], prediction)


#[12000, 7788, 2886, 2433.0881, 2132.6492, 1913.0713, 1739.1803, 1610.0355, 1511.5085, 1437.265, 1378.7538, 1332.4576, 1294.2588]
