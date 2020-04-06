import keras
import numpy as np

data_type = "confirmed"
model_path = "models/20200406-151940/"+data_type+"_model.h5"

#china [27.3338, 128.5697],[56.0, 90.0, 114.0] => 146.55475
#suisse [46.8182, 8.2275],[56.0, 90.0, 114.0] => 146.9167

latlng = np.asarray([27.3338, 128.5697])
last_evolution = np.asarray([56.0, 90.0, 114.0])
input = [[[last_evolution]], [latlng]]

# Recreate the exact same model purely from the file
model = keras.models.load_model(model_path)

prediction = model.predict(input)

print(prediction)

#check whether location as a real influence (just a bit)
for i in range(0, 180, 1):
	for j in range(0, 180, 1):
		latlng = np.asarray([j, i])
		last_evolution = np.asarray([56.0, 90.0, 114.0])
		input = [[[last_evolution]], [latlng]]
		prediction = model.predict(input)

		print([j, i], prediction)
