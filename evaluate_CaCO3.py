import argparse  # package for parsing cli arguments
from datetime import datetime  # for capturing the time 
import csv
from scipy.signal import savgol_filter
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow_addons as tfa  # this is for the R2 metric
import tensorflow_probability as tfp
import joblib

# add lines below if you have configured tensorflow to run with GPU
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# scaler = joblib.load('./scalers/normalizer.pkl')

# the custom metric (has to be here when we load custom model or imported)
class RPD(tf.keras.metrics.Metric):

    def __init__(self, name="rpd", **kwargs):

        super().__init__(name=name, **kwargs)
        self.std = self.add_weight(
            name="std", initializer="zeros"
            )
        self.rmse = self.add_weight(
            name="rmse", initializer="zeros"
            )
        self.total_samples = self.add_weight(
            name="total_samples", initializer="zeros", dtype="int32"
            )

    
    def update_state(self, y_true, y_pred, sample_weight=None):

        num_samples = tf.shape(y_pred)[0]
        self.total_samples.assign_add(num_samples)
        y_mean = tf.reduce_sum((y_true)/tf.cast(self.total_samples, tf.float32))
        std = tf.sqrt(tf.reduce_sum(tf.square(y_true - y_mean))/tf.cast(self.total_samples, tf.float32))
        self.std.assign_add(std)
        rmse = tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred))/tf.cast(self.total_samples, tf.float32))
        self.rmse.assign_add(rmse)


    def result(self):
        return  self.std/self.rmse


    def reset_state(self):        
        self.std.assign(0.)
        self.rmse.assign(0.)
        self.total_samples.assign(0)


# class RPIQR(tf.keras.metrics.Metric):

#     def __init__(self, name="rpiqr", **kwargs):

#         super().__init__(name=name, **kwargs)
#         self.rmse = self.add_weight(
#             name="rmse", initializer="zeros"
#             )
#         self.total_samples = self.add_weight(
#             name="total_samples", initializer="zeros", dtype="int32"
#             )
#         self.iqr = self.add_weight(
#             name="iqr", initializer="zeros", dtype="float32"
#             )

    
#     def update_state(self, y_true, y_pred, sample_weight=None):

#         num_samples = tf.shape(y_pred)[0]
#         self.total_samples.assign_add(num_samples)
#         percentiles = tfp.stats.percentile(y_true, q=[25., 75.])
#         self.iqr.assign_add(percentiles[1]-percentiles[0])
#         rmse = tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred))/tf.cast(self.total_samples, tf.float32))
#         self.rmse.assign_add(rmse)


#     def result(self):
#         return  self.iqr/self.rmse


#     def reset_state(self):        
#         self.iqr.assign(0.)
#         self.rmse.assign(0.)
#         self.total_samples.assign(0)

# capture time now
time_now = datetime.now()
# convert datetime object to string
time_start_string = datetime.strftime(time_now, '%Y%m%dT%H:%M:%S')
# state the cli arg parser
my_parser = argparse.ArgumentParser(description='Make predictions on custom CaCO3 reflectance data.')
# argument input path
my_parser.add_argument('-i',
                       '--input',
                       metavar='input_path',
                       type=str,
                       help='The path to the input data.'
                    )
# argument output path
my_parser.add_argument('-o',
                       '--output',
                       metavar='output_path',
                       type=str,
                       default=f'./CaCO3_results_{time_start_string}.csv',
                       help='The path where the predictions will be saved.'
                    )                    
# argument model path
my_parser.add_argument('-m',
                       '--model',
                       metavar='model_path',
                       type=str,
                    #    default='./model',
                       help='The path where your model is saved.'
                    )   

args = my_parser.parse_args()
# define static variables 
INPUT_PATH = vars(args)['input']
OUTPUT_PATH = vars(args)['output']
MODEL_PATH = vars(args)['model']
real_values = ['Real_values']
pred_values = ['Predicted_values']
# load your data and targets
input_array = np.loadtxt(INPUT_PATH, skiprows=1, delimiter=',', usecols = range(2, 2695, 4), dtype=float)
target_array = np.loadtxt(INPUT_PATH, skiprows=1, delimiter=',', usecols = (1,), dtype=float)
# convert to absorbance
absorbance = np.round(-np.log10(input_array/100), 6)
# Calculate derivatives
absorbance_smoothed = savgol_filter(absorbance, 13, polyorder = 2, deriv=2)

# _ = np.concatenate([absorbance_smoothed, np.expand_dims(target_array, axis=1)], axis=1)
# _ = scaler.transform(_)
# absorbance_smoothed = _[:, :-1]
# target_array = _[:, -1]


# load model
# model = load_model(MODEL_PATH, custom_objects={'RPD': RPD, 'RPIQR': RPIQR})
model = load_model(MODEL_PATH, custom_objects={'RPD': RPD})
# print metrics in terminal
print(model.evaluate(x=absorbance_smoothed, y=target_array, batch_size = 1))
# get prediction values
preds = model.predict(absorbance_smoothed)
# these next lines are for print out reasons
pred_values += list(preds.flatten())
real_values += list(target_array)
rows = zip(real_values, pred_values)

with open(OUTPUT_PATH, 'w') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)
