
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import interp2d
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score
import random
import copy
from numpy.linalg import norm
# Necessary packages
import matplotlib.pyplot as plt

import argparse
import numpy as np

from dataloder import data_loader
from gamcn import gamcn
from utils import rmse_loss
# Just disables the warning, doesn't enable AVX/FMA
import os
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

T = np.loadtxt(r'T.txt', dtype=float)

# print(T.shape)
# (878, 878)D:/G/GAMCNMDF/database/HMDD v3.2\T.txt
miRNA_disease_k = np.loadtxt(r"known.txt", dtype=int)
miRNA = np.loadtxt(r"mirnaxiangsixing.txt", dtype=float)
miRNA_disease_uk = np.loadtxt(r"unknown.txt", dtype=int)


# print(miRNA_disease_k.shape)
# print(miRNA_disease_k.shape)
def main(args):
    '''Main function for UCI letter and spam datasets.

    Args:
      - data_name: letter or spam
      - miss_rate: probability of missing components
      - batch:size: batch size
      - hint_rate: hint rate
      - alpha: hyperparameter
      - iterations: iterations

    Returns:
      - imputed_data_x: imputed data
      - rmse: Root Mean Squared Error
    '''

    miss_rate = args.miss_rate

    gamcn_parameters = {'batch_size': args.batch_size,
                       'hint_rate': args.hint_rate,
                       'alpha': args.alpha,
                       'iterations': args.iterations}

    # Load data and introduce missingness
    ori_data_x, miss_data_x, data_m = data_loader(miss_rate)

    # Impute missing data
    imputed_data_x = gamcn(miss_data_x, gamcn_parameters)

    # Report the RMSE performance
    rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)

    print()
    print('RMSE Performance: ' + str(np.round(rmse, 4)))

    return imputed_data_x, rmse


if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        choices=['letter', 'spam'],
        default='spam',
        type=str)
    parser.add_argument(
        '--miss_rate',
        help='missing data probability',
        default=0.24,
        type=float)
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch',
        default=128,
        type=int)
    parser.add_argument(
        '--hint_rate',
        help='hint probability',
        default=0.9,
        type=float)
    # 个正则化设计,和dropout类似防过拟合的
    parser.add_argument(
        '--alpha',
        help='hyperparameter',
        default=0.5,

        type=float)
    parser.add_argument(
        '--iterations',
        help='number of training interations',
        default=10,
        type=int)

    args = parser.parse_args()

    # Calls main function

  #  np.savetxt('D:/G/20220930/finall/final/database/HMDD v2.0/mdtwo.txt', imputed_data, fmt="%0.10f")


# print(imputed_data)

