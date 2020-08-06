# Machine Learning Capstone Project: Stock Predictor
The aim of this project is to built stock predictor.

In this project I built stock price predictor that takes daily trading data for a list of ticker symbols (e.g.
GOOG, AAPL) over a last 5 years span as input, and outputs the predicted stock prices for each of those stocks for next 30 days.

## Libraries used:
- import numpy as np
- import pandas as pd
import json
import time
from datetime import datetime
from requests import request
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error

##### Sagemaker resources

import boto3
import sagemaker
from sagemaker import get_execution_role
import os
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.estimator import Estimator
from sagemaker.tuner import IntegerParameter, ContinuousParameter, HyperparameterTuner

#Import other Tensorflow libraries

import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM



## Project reference files:
- The project proposal is the proposal.pdf, and the final report is the report.pdf.
- Project code and results can be found in the Stock Predictior Project.ipynb.
