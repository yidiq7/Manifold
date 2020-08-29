from hypersurface_tf import *
from generate_h import *
from biholoNN import *
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

z0, z1, z2, z3, z4 = sp.symbols('z0, z1, z2, z3, z4')
Z = [z0,z1,z2,z3,z4]
f = z0**5 + z1**5 + z2**5 + z3**5 + z4**5 + 0.5*z0*z1*z2*z3*z4
#np.random.seed(123)
HS = Hypersurface(Z, f, 10000)
#np.random.seed(124)
HS_test = Hypersurface(Z, f, 10000)

train_set = generate_dataset(HS)
test_set = generate_dataset(HS_test)

train_set = train_set.shuffle(500000).batch(1000)
test_set = test_set.shuffle(50000).batch(1000)

class KahlerPotential(tf.keras.Model):

    def __init__(self):
        super(KahlerPotential, self).__init__()
        self.biholomorphic = Biholomorphic()
        self.layer1 = Dense(25,500, activation=tf.square)
        self.layer2 = Dense(500,500, activation=tf.square)

    def call(self, inputs):
        x = self.biholomorphic(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        #x = self.layer3(x)
        x = tf.reduce_sum(x, 1)
        x = tf.math.log(x)
        return x

model = KahlerPotential()

@tf.function
def volume_form(x, Omega_Omegabar, mass, restriction):

    kahler_metric = complex_hessian(tf.math.real(model(x)), x)
    volume_form = tf.math.real(tf.linalg.det(tf.matmul(restriction, tf.matmul(kahler_metric, restriction, adjoint_b=True))))
    weights = mass / tf.reduce_sum(mass)
    factor = tf.reduce_sum(weights * volume_form / Omega_Omegabar)
    #factor = tf.constant(35.1774, dtype=tf.complex64)
    return volume_form / factor

optimizer = tf.keras.optimizers.Adam()
epochs = 6000

for epoch in range(epochs):
    for step, (points, Omega_Omegabar, mass, restriction) in enumerate(train_set):
        with tf.GradientTape() as tape:
            
            omega = volume_form(points, Omega_Omegabar, mass, restriction)
            loss = weighted_MAPE(Omega_Omegabar, omega, mass)  
            grads = tape.gradient(loss, model.trainable_weights)

        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if step % 500 == 0:
            print("step %d: loss = %.4f" % (step, loss))
    
    test_loss = 0
    test_loss_old = 100
    total_mass = 0
    
    for step, (points, Omega_Omegabar, mass, restriction) in enumerate(test_set):
        omega = volume_form(points, Omega_Omegabar, mass, restriction)
        mass_sum = tf.reduce_sum(mass)
        test_loss += weighted_MAPE(Omega_Omegabar, omega, mass) * mass_sum
        total_mass += mass_sum
   
    test_loss = test_loss / total_mass
    print("test_loss:", test_loss.numpy())
    
