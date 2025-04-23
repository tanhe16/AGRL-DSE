from utils import *

class GraphConvolutionSparse():

    def __init__(self, input_dim, output_dim, adj, features_nonzero, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
            x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])

            x = tf.sparse_tensor_dense_matmul(self.adj, x)
            outputs = self.act(x)
        return outputs

class GraphSAGE():
    def __init__(self, input_dim, output_dim, adj, name, dropout=0., act=tf.nn.relu, aggregator_type='mean'):
        self.name = name
        self.vars = {}
        self.adj = adj
        self.dropout = dropout
        self.act = act
        self.aggregator_type = aggregator_type
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')
            self.vars['weights_aggregator'] = weight_variable_glorot(input_dim, output_dim, name='weights_aggregator')

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            if isinstance(x, tf.SparseTensor):
                x = tf.sparse_tensor_to_dense(x)
            x = tf.nn.dropout(x, 1-self.dropout)
            neighbors = tf.sparse_tensor_dense_matmul(self.adj, x)
            if self.aggregator_type == 'mean':
                x = tf.add(x, neighbors)
                x = tf.multiply(x, 0.5)
            elif self.aggregator_type == 'sum':
                x = tf.add(x, neighbors)
            elif self.aggregator_type == 'max':
                x = tf.maximum(x, neighbors)
            x = tf.matmul(x, self.vars['weights'])
            outputs = self.act(x)
        return outputs

class GraphAttention():
    def __init__(self, input_dim, output_dim, adj, name, dropout=0., act=tf.nn.elu, alpha=0.2, num_heads=1):
        self.name = name
        self.vars = {}
        self.adj = adj
        self.dropout = dropout
        self.act = act
        self.alpha = alpha
        self.num_heads = num_heads
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')
            self.vars['attention'] = weight_variable_glorot(2 * output_dim, 1, name='attention')

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            if isinstance(x, tf.SparseTensor):
                x = tf.sparse_tensor_to_dense(x)
            x = tf.nn.dropout(x, 1 - self.dropout)
            h = tf.matmul(x, self.vars['weights'])
            N = tf.shape(h)[0]
            f_1 = tf.layers.dense(h, units=1, use_bias=False)
            f_2 = tf.layers.dense(h, units=1, use_bias=False)
            logits = tf.add(f_1, tf.transpose(f_2, [1, 0]))
            coefs = tf.nn.softmax(tf.nn.leaky_relu(logits, alpha=self.alpha), axis=-1)
            coefs = tf.nn.dropout(coefs, 1 - self.dropout)
            vals = tf.matmul(coefs, h)
            outputs = self.act(vals)
        return outputs


class InnerProductDecoder():

    def __init__(self, input_dim, name, num_r, dropout=0., act=tf.nn.sigmoid):
        self.name = name
        self.vars = {}
        self.issparse = False
        self.dropout = dropout
        self.act = act
        self.num_r = num_r

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, input_dim, name='weights')

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            inputs = tf.nn.dropout(inputs, 1-self.dropout)
            R = inputs[0:self.num_r, :]
            D = inputs[self.num_r:, :]
            R = tf.matmul(R, self.vars['weights'])
            D = tf.transpose(D)
            x = tf.matmul(R, D)
            x = tf.reshape(x, [-1])
            outputs = self.act(x)
        return outputs
