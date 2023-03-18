
import numpy as np
from utils import binary_sampler
from keras.datasets import mnist


def data_loader( miss_rate):
  data_x = np.loadtxt(r'T.txt', dtype=float)


  no, dim = data_x.shape


  data_m = binary_sampler(1 - miss_rate, no, dim)
  miss_data_x = data_x.copy()
  miss_data_x[data_m == 0] = np.nan

  return data_x, miss_data_x, data_m



