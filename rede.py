import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


seed = 42

np.random.seed(seed)
tf.random.set_seed(seed)

data=pd.read_csv('combined_keypoints.csv')