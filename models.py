import tensorflow as tf
from biholoNN import *

class onelayer(tf.keras.Model):

    def __init__(self, n_units):
        super(onelayer, self).__init__()
        self.biholomorphic = Biholomorphic()
        self.layer1 = Dense(25, n_units[0], activation=tf.square)
        self.layer2 = Dense(n_units[0], 1)

    def call(self, inputs):
        x = self.biholomorphic(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = tf.math.log(x)
        return x


class twolayers(tf.keras.Model):

    def __init__(self, n_units):
        super(twolayers, self).__init__()
        self.biholomorphic = Biholomorphic()
        self.layer1 = Dense(25, n_units[0], activation=tf.square)
        self.layer2 = Dense(n_units[0], n_units[1], activation=tf.square)
        self.layer3 = Dense(n_units[1], 1)

    def call(self, inputs):
        x = self.biholomorphic(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = tf.math.log(x)
        return x


class threelayers(tf.keras.Model):

    def __init__(self, n_units):
        super(threelayers, self).__init__()
        self.biholomorphic = Biholomorphic()
        self.layer1 = Dense(25, n_units[0], activation=tf.square)
        self.layer2 = Dense(n_units[0], n_units[1], activation=tf.square)
        self.layer3 = Dense(n_units[1], n_units[2], activation=tf.square)
        self.layer4 = Dense(n_units[2], 1)

    def call(self, inputs):
        x = self.biholomorphic(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = tf.math.log(x)
        return x


class fourlayers(tf.keras.Model):

    def __init__(self, n_units):
        super(fourlayers, self).__init__()
        self.biholomorphic = Biholomorphic()
        self.layer1 = Dense(25, n_units[0], activation=tf.square)
        self.layer2 = Dense(n_units[0], n_units[1], activation=tf.square)
        self.layer3 = Dense(n_units[1], n_units[2], activation=tf.square)
        self.layer4 = Dense(n_units[2], n_units[3], activation=tf.square)
        self.layer5 = Dense(n_units[3], 1)

    def call(self, inputs):
        x = self.biholomorphic(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = tf.math.log(x)
        return x


class fivelayers(tf.keras.Model):

    def __init__(self, n_units):
        super(fivelayers, self).__init__()
        self.biholomorphic = Biholomorphic()
        self.layer1 = Dense(25, n_units[0], activation=tf.square)
        self.layer2 = Dense(n_units[0], n_units[1], activation=tf.square)
        self.layer3 = Dense(n_units[1], n_units[2], activation=tf.square)
        self.layer4 = Dense(n_units[2], n_units[3], activation=tf.square)
        self.layer5 = Dense(n_units[3], n_units[4], activation=tf.square)
        self.layer6 = Dense(n_units[4], 1)

    def call(self, inputs):
        x = self.biholomorphic(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = tf.math.log(x)
        return x


