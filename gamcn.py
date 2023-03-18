# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''GAIN function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
'''

# Necessary packages
# import tensorflow as tf
##IF USING TF 2 use following import to still use TF < 2.0 Functionalities
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
from tqdm import tqdm

from utils import normalization, renormalization, rounding
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index


def gain(data_x, gain_parameters):
    '''Impute missing values in data_x

    Args:
      - data_x: original data with missing values
      - gain_parameters: GAIN network parameters:
        - batch_size: Batch size
        - hint_rate: Hint rate
        - alpha: Hyperparameter
        - iterations: Iterations

    Returns:
      - imputed_data: imputed data
    '''
    # Define mask matrix
    data_m = 1 - np.isnan(data_x)
    # np.isnan()是判断是否是空值,返回false或者true

    # System parameters
    batch_size = gain_parameters['batch_size']
    hint_rate = gain_parameters['hint_rate']
    alpha = gain_parameters['alpha']
    iterations = gain_parameters['iterations']

    # Other parameters
    no, dim = data_x.shape

    # Hidden state dimensions，数据的列数
    h_dim = int(dim)

    # Normalization
    norm_data, norm_parameters = normalization(data_x)
    # 把缺失数据的值设置成0
    norm_data_x = np.nan_to_num(norm_data, 0)

    # 下面我们构建 X, M, H 在模型中的占位，行数不定，列数为dim
    # GAIN architecture
    # Input placeholders
    # Data vector
    X = tf.placeholder(tf.float32, shape=[None, dim])
    # Mask vector
    M = tf.placeholder(tf.float32, shape=[None, dim])
    # Hint vector
    H = tf.placeholder(tf.float32, shape=[None, dim])
    # placeholder此函数可以理解为形参,用于定义过程,在执行的时候再赋具体的值。
    # 不必指定初始值,可在运行时,通过 Session.run 的函数的 feed_dict 参数指定
    # Discriminator variables，开始初始化判别器中的参数

    # D_W1行数为dim*2，列数为h_dim。
    D_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))  # Data + Hint as inputs
    # D_b1是一个大小为h_dim的零向量。
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    # xavier_init 是一个通过该层的输入和输出参数个数得到的分布范围内的均匀分布
    D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))
    # xavier_init,尽可能的让输入和输出服从相同的分布，这样就能够避免后面层的激活函数的输出值趋向于0。
    D_W3 = tf.Variable(xavier_init([h_dim, dim]))
    D_b3 = tf.Variable(tf.zeros(shape=[dim]))  # Multi-variate outputs
    '''D_W1的行数是其他行数的两倍的原因是我们需要原始数据和hint_matrix作为我们的输入值。
  theta_D表示判别器可训练的参数集合。在这里为[D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]'''
    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    # Generator variables
    # Data + Mask as inputs (Random noise is in missing components)
    # 接下来定义生成器参数：
    G_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    '''G_W1的行数是其他行数的两倍的主要原因是我们需要原始数据和mask_matrix作为我们的输入值。
  theta_G表示生成器可训练的参数集合。在这里为[G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]'''
    G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W3 = tf.Variable(xavier_init([h_dim, dim]))
    G_b3 = tf.Variable(tf.zeros(shape=[dim]))

    G_W4 = tf.Variable(xavier_init([h_dim, dim]))
    G_b4 = tf.Variable(tf.zeros(shape=[dim]))

    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    ## GAIN functions
    # Generator,定义生成器：
    def generator(x, m):
        # Concatenate Mask and Data
        # 级联掩码和数据  连接,# 将 data 和 mask_matrix 合并
        inputs = tf.concat(values=[x, m], axis=1)
        # G_h1 = f(inputs*G_W1 + G_b1)
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
        # G_h2 = f(G_h1*G_W2 + G_b2
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
        G_h3 = tf.nn.relu(tf.matmul(G_h2, G_W3) + G_b3)
        # MinMax normalized output
        # G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
        G_prob = tf.nn.sigmoid(tf.matmul(G_h3, G_W4) + G_b4)  # D_logit经过一个sigmoid函数，D_prob维度为[N, 1]

        return G_prob

    # Discriminator定义判别器：
    def discriminator(x, h):
        # Concatenate Data and Hint, # 将 data 和 hint_matrix 合并
        inputs = tf.concat(values=[x, h], axis=1)
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_logit = tf.matmul(D_h2, D_W3) + D_b3
        D_prob = tf.nn.sigmoid(D_logit)
        # D_prob ---> m_hat = D(x_hat, h)
        return D_prob

    ## GAIN structure
    # Generator, # G_sample ---> x_bar = G(x_tilde,m)
    G_sample = generator(X, M)

    # Combine with observed data
    # Hat_X    ---> x_hat = m * x_tilde + (1-m) * x_hat
    Hat_X = X * M + G_sample * (1 - M)

    # Discriminator
    # D_prob   ---> m_hat = D(x_hat, h)
    D_prob = discriminator(Hat_X, H)

    ## GAIN loss对应的是Pseudo-code里面的部分 得到x横和x尖
    D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) \
                                  + (1 - M) * tf.log(1. - D_prob + 1e-8))

    G_loss_temp = -tf.reduce_mean((1 - M) * tf.log(D_prob + 1e-8))

    MSE_loss = \
        tf.reduce_mean((M * X - M * G_sample) ** 2) / tf.reduce_mean(M)

    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss

    ## GAIN solver
    # D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    # G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
    D_solver = tf.train.MomentumOptimizer(learning_rate=0.0001, momentum=0.5).minimize(D_loss, var_list=theta_G)
    G_solver = tf.train.MomentumOptimizer(learning_rate=0.0001, momentum=0.5).minimize(G_loss, var_list=theta_G)
    # tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.5).minimize(G_loss, var_list=theta_G)
    ## Iterations
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Start Iterations
    for it in tqdm(range(iterations)):
        # Sample batch
        batch_idx = sample_batch_index(no, batch_size)
        X_mb = norm_data_x[batch_idx, :]
        M_mb = data_m[batch_idx, :]
        # Sample random vectors
        Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
        # Sample hint vectors
        H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
        H_mb = M_mb * H_mb_temp

        # Combine random vectors with observed vectors
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        _, D_loss_curr = sess.run([D_solver, D_loss_temp],
                                  feed_dict={M: M_mb, X: X_mb, H: H_mb})
        _, G_loss_curr, MSE_loss_curr = \
            sess.run([G_solver, G_loss_temp, MSE_loss],
                     feed_dict={X: X_mb, M: M_mb, H: H_mb})

    ## Return imputed data
    Z_mb = uniform_sampler(0, 0.01, no, dim)
    M_mb = data_m
    X_mb = norm_data_x
    X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

    imputed_data = sess.run([G_sample], feed_dict={X: X_mb, M: M_mb})[0]

    imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data

    # Renormalization
    imputed_data = renormalization(imputed_data, norm_parameters)

    # Rounding
    imputed_data = rounding(imputed_data, data_x)

    return imputed_data
