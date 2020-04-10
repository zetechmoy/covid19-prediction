# Covid19-Prediction

Covid19-Prediction is a simple web app which predict the next covid19 evolution in function of location and last evolution. This is a project based on AI using Keras and served by Flask.

>Prediction are for science purposes only.

![Example Of Website](https://i.imgur.com/IRkmQdx.png)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.

```bash
pip3 install -r reqs.txt
```

## Usage
- Start the app :
```bash
python3 run.py
```
- Open your browser on [http://127.0.0.1:5000](http://127.0.0.1:5000)
- Then follow instructions on the website

## Train
This app come with 3 trained models. They have been trained on 09/04/2020.
If you want to train your own data or update models with actual values simply run
```bash
cd data/ ; sh update.sh ; cd ..; echo "Done !"
```
Data comes from : [CSSEGISandData/COVID-19](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series)

Then train recovered model
```bash
python3 train.py -dt recovered
```
"recovered" can be replaced by "deaths" or "confirmed" to train corresponding model.

Need help ?
```bash
python3 train.py --help
```

## Test
Test your model using
```bash
python3 test.py path/to/your/model.h5
```
Sample data and more advanced tests are in test.py

## Authors

* **Théo Guidoux** - [Github](https://github.com/zetechmoy) - [Twitter/@TGuidoux](https://twitter.com/TGuidoux)

## License
[MIT](https://choosealicense.com/licenses/mit/)
