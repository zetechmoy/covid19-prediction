
from helper import *

#define which data we are working on : confirmed, deaths or recovered
data_type = "confirmed"

def main():
	data = get_data("data/time_series_covid19_"+data_type+"_global.csv")
	print(data)


if __name__ == '__main__':
	main()
