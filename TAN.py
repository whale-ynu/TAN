import copy

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
np.set_printoptions(suppress=True)
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
import math

from tensorflow.python.layers.normalization import batch_norm
#from tensorflow.contrib.layers import xavier_initializer
import loading
from minibatch import MinibatchIterator





rate=5
model_id = 565
MODEL_SAVE_PATH = "Model/model_0"
MODEL_NAME = "Qos_0"

def parse_args():
    parser = argparse.ArgumentParser(description='run TAN')
    parser.add_argument('--training', default=False, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lamda_bilinear', default=1.0, type=float)
    parser.add_argument('--hidden_lstm', default=64, type=int)
    parser.add_argument('--input_feature', default=64, type=int)
    parser.add_argument('--hidden_predict', default=64, type=int)
    parser.add_argument('--num_nodes', default=300000, type=int)
    parser.add_argument('--regular', default=False, type=bool)
    parser.add_argument('--optimizer', nargs='?', default='AdamOptimizer')
    parser.add_argument('--length_conv', default=6, type=int)
    parser.add_argument('--dropout', default=0.0001, type=float)
    parser.add_argument('--layer_num', default=2, type=int)
    return parser.parse_args()


class model(object):

    def __init__(self, training, epochs,hidden_lstm,learning_rate,batch_size,lamda_bilinear,input_feature,num_nodes,regular,optimizer,
                 Short_Path,largest_path,hidden_predict,length_conv,dropout,layer_num):

        self.random_seed=2020
        self.training=training
        self.epochs=epochs
        self.hidden_lstm=hidden_lstm
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.lamda_bilinear=lamda_bilinear
        self.input_feature=input_feature
        self.num_nodes=num_nodes
        self.regular=regular
        self.optimizer_type=optimizer
        self.largest_path=largest_path
        self.Short_Path=Short_Path
        self.hidden_predict=hidden_predict
        self.length_conv=length_conv
        self.dropout=dropout
        self.layer_num=layer_num
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)

            # Input data.
            self.contruct_placeholder()


            # prediction
            self.y = self.forward(self.largest_path)

            # compute the loss.
            self.loss = tf.reduce_sum(tf.abs(tf.subtract(self.input_y, self.y)))
            #self.loss = tf.reduce_mean(tf.square(tf.subtract(self.input_y, self.y)))

            # regular
            if (self.regular):
                self.loss = self.loss +tf.contrib.layers.l2_regularizer(self.lamda_bilinear)( self.weights['layer_0'])+tf.contrib.layers.l2_regularizer(self.lamda_bilinear)(self.weights['layer_1'])
            # Optimizer.

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                if self.optimizer_type == 'AdamOptimizer':
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                            epsilon=1e-8).minimize(self.loss)
                elif self.optimizer_type == 'AdagradOptimizer':
                    self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                               initial_accumulator_value=1e-8).minimize(self.loss)
                elif self.optimizer_type == 'GradientDescentOptimizer':
                    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                        self.loss)
                elif self.optimizer_type == 'MomentumOptimizer':
                    self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                        self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
            
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            else:
                print('No Checkpoint Found')

    def BiRNN(self,x,length_path):
        # prepare data shape to match bidirectional rnn function requirements
        # current data input shape: (batch_size, n_steps, n_input)
        # required shape: n_steps tensors list of shape(batch_size, n_input)
        # unstack to get a list of n_steps tensors of shape (batch_size, n_input)

        # forward
        lstm_fw_cell = rnn.BasicLSTMCell(self.input_feature, forget_bias=0.6)
        # backword
        lstm_bw_cell = rnn.BasicLSTMCell(self.input_feature, forget_bias=0.6)
        #bi_lstm
        output, output_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, sequence_length=length_path,dtype=tf.float32)
        out = tf.concat([output_state[0][-1], output_state[1][-1]],axis=-1)     #[batch_size,length,2*hidden_lstm]
        return out



    def get_weights_bias(self):
        self.weights = dict()
        self.bias = dict()
        for i in range(self.layer_num):
            if i==0:
                self.weights['layer_%d' % i] = tf.Variable(
                    tf.random_normal([self.hidden_lstm*2, self.hidden_predict]))
                self.bias['layer_%d' % i] = tf.Variable(tf.random_normal([self.hidden_predict]))
            elif i==self.layer_num-1:
                self.weights['layer_%d' % i] = tf.Variable(tf.random_normal([self.hidden_predict, 1]))
                self.bias['layer_%d' % i] = tf.Variable(tf.random_normal([1]))
            else:
                self.weights['layer_%d'%i]=tf.Variable(tf.random_normal([self.hidden_predict, self.hidden_predict]))
                self.bias['layer_%d'%i]=tf.Variable(tf.random_normal([self.hidden_predict]))

        glorot = np.sqrt(2.0 / (self.hidden_lstm + self.hidden_predict))
        self.weights['fusion_0'] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(2 * self.hidden_lstm, 2 * self.hidden_predict)),dtype=np.float32)
        self.weights['fusion_1'] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(2 * self.hidden_lstm, 2 * self.hidden_predict)),dtype=np.float32)

        return self.weights, self.bias



    def batch_norm_layer(self, x, train_phase):
        bn_train = batch_norm(x, training=True)
        bn_inference = batch_norm(x, training=False)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z



    def conv(self, name, inputs, k_size, nums_in, nums_out, strides):
        # init kernel
        kernel = tf.get_variable(name + "W", [k_size, k_size, nums_in, nums_out],
                                 initializer=tf.truncated_normal_initializer(stddev=0.01))
        # init bias
        bias = tf.get_variable(name + "B", [nums_out], initializer=tf.constant_initializer(0.))

        # conv with same
        input = tf.nn.conv2d(
            inputs, kernel, [1, strides, strides, 1], "SAME") + bias

        return input


    def forward(self,largest_path):

        self.largest_path=largest_path
        self.get_weights_bias()
        #----------------------------------------Input Layer------------------------------------------------------------
        #init path node
        self.input_embedding = tf.Variable(
            tf.truncated_normal([self.num_nodes, self.input_feature], stddev=0.1), name='input_embedding')
        self.input_bias = tf.Variable(
            tf.truncated_normal([self.num_nodes],mean=0, stddev=0.1), name='input_bias')
        # use vector of zero completing the path can't reach the longest
        self.add = tf.Variable(tf.zeros([1, self.input_feature]))
        self.input_embedding= tf.concat([self.input_embedding, self.add],0)
        self.input=tf.reshape(self.input_path,[self.batch_size*self.largest_path])
        self.feature_pathnode = tf.nn.embedding_lookup(self.input_embedding,self.input_path)  # [batchsize,path_length,input_feature]
        self.feature_pathnode = tf.reshape(self.feature_pathnode, [self.batch_size, self.largest_path, self.input_feature])
        #init user/service/uas/sas
        self.user_feature = tf.nn.embedding_lookup(self.input_embedding, self.input_user)  # [batchsize,hidden_feature]
        self.service_feature = tf.nn.embedding_lookup(self.input_embedding, self.input_service)
        self.uas_feature = tf.nn.embedding_lookup(self.input_embedding, self.input_uas)
        self.sas_feature = tf.nn.embedding_lookup(self.input_embedding, self.input_sas)

        #------------------------------------------Explicit Path Layer--------------------------------------------------
        self.path_feature=self.BiRNN(self.feature_pathnode,self.length_path)  #[batchsize,10,input_feature*2]

        if self.dropout:
         self.path_feature=tf.nn.dropout(self.path_feature, 1-self.dropout)


         # ------------------------------------------Implicit End-cross Layer-------------------------------------------


        self.feature_origin=tf.squeeze(self.user_feature,1)
        self.feature_desitination=tf.squeeze(self.service_feature,1)
        #out_product
        self.inputs = tf.einsum('ij,ik->ijk',self.feature_origin,self.feature_desitination)
        self.inputs = tf.expand_dims(self.inputs, -1)

        #conv
        for l in range(self.length_conv):
            if (l == 0):
                self.inputs = self.conv('conv%d' % l, self.inputs, 2, 1, self.input_feature*2, 2)
            else:
                self.inputs = self.conv('conv%d' % l, self.inputs, 2, 16, self.input_feature*2, 2)
            self.inputs = tf.nn.relu(self.inputs)     #[256,1,1,128]
            if self.dropout:
                self.inputs = tf.nn.dropout(self.inputs,1-self.dropout)
                
        self.feature_node=tf.squeeze(self.inputs,[1,2])

        #------------------------------------------Gating Layer---------------------------------------------------------
        self.f_0 = tf.matmul(self.feature_node, self.weights['fusion_0'])
        self.f_1 = tf.matmul(self.path_feature, self.weights['fusion_1'])
        self.node = tf.add(self.f_0, self.f_1)
        #user = self.batch_norm_layer(self.node, train_phase=self.train_phase)
        self.G = tf.nn.sigmoid(self.node)
        self.input_predict = self.G * self.feature_node + (1 - self.G) *self.path_feature
        if self.dropout:
            self.input_predict=tf.nn.dropout(self.input_predict,1-self.dropout)

        #-------------------------------------------Pridiction Layer----------------------------------------------------
        for i in range(self.layer_num):
            if i==0:
                self.result=self.input_predict
            if i!=self.layer_num-1:
                self.result= tf.nn.relu(tf.matmul(self.result,self.weights['layer_%d'%i])+self.bias['layer_%d'%i])
            else:
                self.result = tf.matmul(self.result, self.weights['layer_%d' % i]) + self.bias[
                    'layer_%d' % i]

        return self.result


    def contruct_placeholder(self):
        self.input_path = tf.placeholder(tf.int32,shape=(self.batch_size,self.largest_path),name='input_path')
        self.input_y = tf.placeholder(tf.float32, shape=(self.batch_size,1), name='input_y')
        self.input_user=tf.placeholder(tf.int32, shape=(self.batch_size,1), name='input_user')
        self.input_service = tf.placeholder(tf.int32, shape=(self.batch_size,1), name='input_service')
        self.input_uas=tf.placeholder(tf.int32, shape=(self.batch_size,1), name='input_uas')
        self.input_sas = tf.placeholder(tf.int32, shape=(self.batch_size,1), name='input_sas')
        self.length_path = tf.placeholder(tf.int64, shape=(self.batch_size), name='length_path')
        self.train_phase = tf.placeholder(tf.bool)


    def train(self, train_file,test_file):


        for i in range(self.epochs):

            rmse, mae, nmae = self.evaluate_train(train_file)

            print("Train_%d: RMSE: %f MAE: %f NMAE: %f" % (i, rmse, mae, nmae))
            '''
            rmse1, mae1, nmae1 = self.evaluate_test(test_file)

            print("Test_%d: RMSE: %f MAE: %f NMAE: %f" % (i, rmse1, mae1, nmae1))
            '''





            

    def evaluate_train(self, train_file):
        
        minibatch = MinibatchIterator(train_file, self.batch_size)
        minibatch.generate_batch()
        all_predictions = list()
        all_y = list()
        while not minibatch.is_empty():
            X_U, X_UAS, X_S, X_SAS, Y = minibatch.next_batch()
            self.path_node =[]
            self.length=[]
            for i in range(self.batch_size):
                tmp = copy.deepcopy(Short_Path[str(X_U[i]) + ' ' + str(X_S[i])])
                self.length.append(len(tmp))
                if(len(tmp)<self.largest_path):
                    for i in range(self.largest_path-len(tmp)):
                        tmp.append(self.num_nodes)
                self.path_node.extend(tmp)
            self.path_node=np.array(self.path_node)
            feed_dict_train = {
                self.input_path:np.reshape(self.path_node,[self.batch_size,self.largest_path]),
                self.input_y: np.reshape(Y, [self.batch_size, 1]),
                self.length_path:np.reshape(self.length, [self.batch_size]),
                self.input_user:np.reshape(X_U, [self.batch_size, 1]),
                self.input_service: np.reshape(X_S, [self.batch_size, 1]),
                self.input_uas:np.reshape(X_UAS, [self.batch_size, 1]),
                self.input_sas:np.reshape(X_SAS, [self.batch_size, 1]),
                self.train_phase: True
            }

            _, predictions= self.sess.run(
                    (self.optimizer, self.result),feed_dict=feed_dict_train)

            num_example = len(Y)
            predictions_bounded = np.maximum(np.squeeze(predictions),
                                             np.ones(num_example) * min(Y))  # bound the lower values
            predictions_bounded = np.minimum(predictions_bounded,
                                             np.ones(num_example) * max(Y))  # bound the higher values
            all_y.extend(Y)
            all_predictions.extend(predictions_bounded)

        RMSE = math.sqrt(mean_squared_error(all_predictions, all_y))
        MAE = mean_absolute_error(all_predictions, all_y)
        NMAE = MAE / (max(all_y) - min(all_y))
        return RMSE, MAE, NMAE

    def test(self, test_file):
        rmse, mae, nmae = self.evaluate_test(test_file)
        print("Test: RMSE: %f MAE: %f NMAE: %f" % (rmse, mae, nmae))


    def evaluate_test(self, test_file):
        minibatch = MinibatchIterator(test_file, self.batch_size)
        minibatch.generate_batch()
        all_predictions = list()
        all_y = list()
        while not minibatch.is_empty():
            X_U, X_UAS, X_S, X_SAS, Y = minibatch.next_batch()
            self.path_node = []
            self.length = []
            for i in range(self.batch_size):
                tmp = copy.deepcopy(Short_Path[str(X_U[i]) + ' ' + str(X_S[i])])
                self.length.append(len(tmp))
                if (len(tmp) < self.largest_path):
                    for i in range(self.largest_path - len(tmp)):
                        tmp.append(self.num_nodes)
                self.path_node.extend(tmp)
            self.path_node = np.array(self.path_node)
            feed_dict_test = {
                self.input_path: np.reshape(self.path_node, [self.batch_size, self.largest_path]),
                self.input_y: np.reshape(Y, [self.batch_size, 1]),
                self.length_path: np.reshape(self.length, [self.batch_size]),
                self.input_user: np.reshape(X_U, [self.batch_size, 1]),
                self.input_service: np.reshape(X_S, [self.batch_size, 1]),
                self.input_uas:np.reshape(X_UAS, [self.batch_size, 1]),
                self.input_sas:np.reshape(X_SAS, [self.batch_size, 1]),
                self.train_phase: False
            }
            predictions = self.sess.run((self.y), feed_dict=feed_dict_test)
            num_example = len(Y)
            predictions_bounded = np.maximum(np.squeeze(predictions),
                                             np.ones(num_example) * min(Y))  # bound the lower values
            predictions_bounded = np.minimum(predictions_bounded,
                                             np.ones(num_example) * max(Y))  # bound the higher values
            all_y.extend(Y)
            all_predictions.extend(predictions_bounded)

        RMSE = math.sqrt(mean_squared_error(all_predictions, all_y))
        MAE = mean_absolute_error(all_predictions, all_y)
        NMAE = MAE / (max(all_y) - min(all_y))
        return  RMSE, MAE, NMAE


if __name__ == '__main__':
    args = parse_args()
    longest, Short_Path= loading.load()
    longest=longest
    train_file='data_demo/train.txt'
    test_file='data_demo/test.txt'
    BRNN = model(args.training, args.epochs, args.hidden_lstm, args.learning_rate, args.batch_size,
                 args.lamda_bilinear,args.input_feature, args.num_nodes, args.regular, args.optimizer, Short_Path, 10,
                 args.hidden_predict, args.length_conv,args.dropout, args.layer_num)
    BRNN.train(train_file, test_file)
    BRNN.test(test_file)

    print("successful")
















































