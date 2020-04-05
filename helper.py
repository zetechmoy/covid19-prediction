import numpy as np
import pandas as pd


def get_data(csv_path):
	#data = np.genfromtxt(csv_path, delimiter=',')
	df = pd.read_csv(csv_path, sep=',')
	df = df.replace(np.nan, '', regex=True)
	return np.asarray(df)
