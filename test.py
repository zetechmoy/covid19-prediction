import keras
import numpy as np
import matplotlib.pyplot as plt

data_type = "confirmed"
model_path = "models/20200407-225637/"+data_type+"_model.h5"

#china [27.3338, 128.5697],[56.0, 90.0, 114.0] => 146.55475
#suisse [46.8182, 8.2275],[56.0, 90.0, 114.0] => 146.9167

def draw(xs, ys, color='blue'):
	for i in range(0, min(len(xs), len(ys))-1):
		plt.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], color=color)

#test on simple data
all_values = [56.0, 90.0, 114.0]
latlng = np.asarray([46.131357,-2.4364346])
duration = 15

# Recreate the exact same model purely from the file
model = keras.models.load_model(model_path)

#input = [[[all_values]], [latlng], [duration]]
#prediction = model.predict(input)[0][0]
#
#print(input, "=>", prediction)

#check whether location as a real influence (just a bit)
#for i in range(0, 180, 1):
#	for j in range(0, 180, 1):
#		latlng = np.asarray([j, i])
#		last_evolution = np.asarray([56.0, 90.0, 114.0])
#		input = [[[last_evolution]], [latlng]]
#		prediction = model.predict(input)
#
#		print([j, i], prediction)

#check whether duration as a real influence (just a bit)
#for i in range(0, 60):
#	last_evolution = np.asarray(all_values)
#	input = [[[last_evolution]], [latlng], [i]]
#	prediction = model.predict(input)
#
#	print(last_evolution, latlng, i, "=>", prediction)

#compare predicted and true evolution
#feb 15 2020 => 1580515200

true_ys = [12,12,12,12,12,12,12,12,12,12,14,18,38,57,100,130,191,204,285,377,653,949,1126,1209,1784,2281,2281,3661,4469,4499,6633,7652,9043,10871,12612,14282,16018,19856,22304,25233,29155,32964,37575,40174,44550,52128,56989,59105,64338,89953]

start_x = [12, 12, 12]
start_date = 1580515200
start_duration = 24

xs = list()
ys = list()
latlng = np.asarray([46.131357,-2.4364346])#france
last_evolution = np.asarray(start_x)

#see evolution
for i in range(0, len(true_ys)):
	input = [[[last_evolution]], [latlng], [start_duration]]

	prediction = model.predict(input)[0][0]
	next_last_evolution = np.hstack((last_evolution[1:], [prediction]))
	last_evolution = next_last_evolution

	ys.append(prediction)
	xs.append(start_date)

	start_date += 86400
	start_duration += 1

print(xs)
print(ys)

draw(xs, ys, color="blue")
draw(xs, true_ys, color="green")
plt.show()

#on se rend compte que lorsqu'il s'agit de prédire la valeur suivante il n'y a
#pas de problème mais à long terme les valeurs sont trop fausses (en faisant l'évolution)
