
data_type = "confirmed"
model_path = "models//"+data_type+"_model.h5"

input = [34, 35, 36]

# Recreate the exact same model purely from the file
model = keras.models.load_model(model_path)

prediction = model.predict([input])

print(prediction)
