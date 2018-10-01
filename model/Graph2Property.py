import tensorflow as tf
import numpy as np
import blocks

class Graph2Property():
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.batch_size = FLAGS.batch_size
        self.A = tf.placeholder("float64", shape = [self.batch_size, 50, 50])
        self.X = tf.placeholder("float64", shape = [self.batch_size, 50, 58])
        self.P = tf.placeholder("float64", shape = [self.batch_size])

        self.create_network()

    def create_network(self):
        self.A = tf.cast(self.A, tf.float64)
        self.X = tf.cast(self.X, tf.float64)
        self.P = tf.cast(self.P, tf.float64)
        self.Z = None
        self._X = None
        self._P = None
        latent_dim = self.FLAGS.latent_dim
        num_layers = self.FLAGS.num_layers

        if( self.FLAGS.model == 'GCN' ):
            self._X = blocks.encoder_gcn(self.X, self.A, num_layers)
        elif( self.FLAGS.model == 'GCN+a' ):
            self._X = blocks.encoder_gat(self.X, self.A, num_layers)
        elif( self.FLAGS.model == 'GCN+g' ):
            self._X = blocks.encoder_gcn_gate(self.X, self.A, num_layers)
        elif( self.FLAGS.model == 'GCN+a+g' ):
            self._X = blocks.encoder_gat_gate(self.X, self.A, num_layers)
        elif( self.FLAGS.model == 'GGNN' ):
            self._X = blocks.encoder_ggnn(self.X, self.A, num_layers)

        if (self.FLAGS.readout == 'atomwise' ):
            self.Z, self._P = blocks.readout_atomwise(self._X, latent_dim)
        elif (self.FLAGS.readout == 'graph_gather' ):
            self.Z, self._P = blocks.readout_atomwise(self.X, self._X, latent_dim)

        self.loss = self.calLoss(self.P, self._P)

        self.lr = tf.Variable(0.0, trainable = False)
        self.opt = self.optimizer( self.lr, self.FLAGS.optimizer )
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.saver = tf.train.Saver()
        tf.train.start_queue_runners(sess=self.sess)
        print ("Network Ready")

    def calLoss(self, P, _P):
        batch_size = int(P.get_shape()[0])
        P = tf.reshape(P, [batch_size, -1])
        P = tf.cast(P, tf.float64)
        _P = tf.reshape(_P, [batch_size, -1])
        _P = tf.cast(_P, tf.float64)
        loss = tf.reduce_mean(tf.pow((P-_P),2))

        return loss

    def optimizer(self, lr, opt_type):
        optimizer = None
        if( opt_type == 'Adam' ):
            optimizer = tf.train.AdamOptimizer(lr)
        elif( opt_type == 'RMSProp' ):
            optimizer = tf.train.RMSPropOptimizer(lr)
        elif( opt_type == 'SGD' ):
            optimizer = tf.train.GradientDescentOptimizer(lr)

        return optimizer.minimize(self.loss)

    def get_output(self):
        return self._P, self.loss

    def train(self, A, X, P):
        opt, loss = self.sess.run([self.opt, self.loss], feed_dict = {self.A : A, self.X : X, self.P : P})
        return loss

    def test(self, A, X, P):
        _P, loss = self.sess.run([self._P, self.loss], feed_dict = {self.A : A, self.X : X, self.P : P})
        return _P, loss

    def predict(self, A, X):
        _P = self.sess.run([self._P], feed_dict = {self.A : A, self.X : X})
        return _P

    def get_nodes(self, A, X):
        return self.sess.run(self._X, feed_dict = {self.A : A, self.X : X})

    def get_adjacency(self, A, X):
        return self.sess.run(self._A, feed_dict = {self.A : A, self.X : X})

    def get_attention(self, A, X):
        return self.sess.run(self._A, feed_dict = {self.A : A, self.X : X})

    def get_gates(self, A, X):
        return self.sess.run(self.gates, feed_dict = {self.A : A, self.X : X})

    def get_latent_vector(self, A, X):
        return self.sess.run(self.Z, feed_dict = {self.A : A, self.X : X})

    def generate_molecule(self, Z):
        return self.sess.run(self._P, feed_dict={self.Z : Z}) 

    def save(self, ckpt_path, global_step):
        self.saver.save(self.sess, ckpt_path, global_step=global_step)
        print("model saved to '%s'" % (ckpt_path))
    
    def restore(self, ckpt_path):
        self.saver.restore(self.sess, ckpt_path)

    def assign_lr(self, learning_rate):
        self.sess.run(tf.assign(self.lr, learning_rate))

