#!/usr/bin/env python3
""" WGAN for circle generation """

# third party imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from scipy.optimize import curve_fit

## Training seting
# batch size
mb_size = 64

# learning rate
lr = 1e-3

# Number of neurals
h_dim1 = 128

# Number of epoch
epoch = 200

# constraint epsilon = cons_value^2
cons_value = 0

# latent variables
Noise_dim = 30 

# number of auxiliary variable
con_dim = 1 

# The number of points of circle
num_point = 100

# dimension of data
targ_dim = num_point*2 

# channel of circle data
n = 2

# weight of penalty term 
lam_constraint = 0

t = np.linspace(0, 2*np.pi, num_point)

## function
# the function for fit
def fun_fit(t, r, w):
    '''
    The function is to fit trigonometirx function
    Inputs:
        t -the parameteric
        r -the amplification coefficient
        w -the initial angel
    '''
    return r*np.cos(t+w)

def plot_(data, n, num_point):
    '''
    Plot the n*(n-1) projection of n-dimensional data
    4*4 plots are presented
    Inputs:
        data      - data of cirlce
        n         - channel of the data
        num_point -inumber of points
    '''
    for i in range(n-1):
        for j in range(i+1, n):
            print('figure x',i+1,'and x',j+1)
            gs = gridspec.GridSpec(4, 4)
            gs.update(wspace=0.05, hspace=0.05)
            for k, sample in enumerate(data):
                ax = plt.subplot(gs[k])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                a = np.reshape(sample, (n, num_point))
                ax.plot(a[i], a[j], 'ko', markersize=0.5)
                ax.set_xlim(-1.05, 1.05)
                ax.set_ylim(-1.05, 1.05)
            namesave = 'CircleTraining.pdf'
            plt.savefig(namesave)
            plt.show()
    
def plot_cGANs(data, n, real_data, num_point, name=None):
    '''
    Plot the generated data of cGAN and real data
    Inputs:
        data      - data of cirlce
        n         - channel of the data
        real_data - real data
        num_point - inumber of points
        name      - name of file for saving
    '''
    for i in range(n-1):
        for j in range(i+1, n):
            print('figure x',i+1,'and x',j+1)
            gs = gridspec.GridSpec(4, 4)
            gs.update(wspace=0.05, hspace=0.05)
            for k, sample in enumerate(data):
                if k > 15:
                    break
                ax = plt.subplot(gs[k])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                a = np.reshape(sample, (n, num_point))
                b = np.reshape(real_data[k], (n, num_point))
                if epoch == 200:
                    ax.plot(b[i], b[j], 'ko', markersize=0.5)
                if lam_constraint == 0:
                    ax.plot(a[i], a[j], 'bo', markersize=1)
                else:
                    ax.plot(a[i], a[j], 'ro', markersize=1)
                ax.set_xlim(-1.05, 1.05)
                ax.set_ylim(-1.05, 1.05)
            if name != None:
                namesave = name+'.pdf'
                plt.savefig(namesave)
            plt.show()
            
def plot_cGANs_ring(data, n, real_data, r, name=None):
    '''
    Plot the soft constraints of constrained GANs
    Inputs:
        data      - data of cirlce
        n         - channel of the data
        real_data - real data
        num_point - inumber of points
        r         - radias
        name      - name of file for saving
    '''
    for i in range(n-1):
        for j in range(i+1, n):
            print('figure x',i+1,'and x',j+1)
            gs = gridspec.GridSpec(2, 2)
            #gs.update(wspace=0.02, hspace=0.02)
            for k, sample in enumerate(data):
                if k > 3:
                    break
                ax = plt.subplot(gs[k])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                a = np.reshape(sample, (n, num_point))
                b = np.reshape(real_data[k], (n, num_point))
                ax.plot(a[i], a[j], 'ro' , markersize=2)
                ax.plot(b[i], b[j], 'ko', markersize=2)
                patches = [Wedge((.0, .0), r[k]+cons_value, 0, 360, width=2*cons_value)]
                p = PatchCollection(patches, alpha=0.3, color='g')
                ax.add_collection(p)
                ax.set_xlim(-1.05, 1.05)
                ax.set_ylim(-1.05, 1.05)
            if name != None:
                namesave = name+'.pdf'
                plt.savefig(namesave)
            plt.show()

def generate_samples(num_samp, num_point, t=t):
    '''
    This function is used to generate high dimensional ellipse samples
    Input:
        num_samp  -the number of samples 
        n         -the dimension of ellipse
        num_point -the dimenson of hyperparameter
    Outputs:
        circle    -the training data
        label     -the radias of training data
    '''
    circle = []
    label = np.random.uniform(low=0.4, high=0.8, size=(num_samp, 1))
    
    for i, a in enumerate(label):
        x = a*np.cos(t)
        y = a*np.sin(t)
        data = [x, y]
        data = np.reshape(data, num_point*2)
        circle.append(data)
    return np.asarray(circle), np.asarray(label)

def xavier_init(size):
    '''
    Initialize the weights for neural network
    Inputs:
        size   - the size of weigts
    Outputs:
        -the initial weights
    '''
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def log(x):
    return tf.log(x + 1e-8)

def lrelu(x):
    '''
    The LReLu activate function of neural network
    '''
    return tf.maximum(x, tf.multiply(x, 0.2))

def sample_Z(m, n):
    '''
    Generate latent variables
    '''
    return np.random.randn(m, n)

# plot training samples
circle, label = generate_samples(16, num_point)
plot_(circle, 2, num_point)

#-----------------------------------------------------------------------------#
# placeholders of discriminator
target = tf.placeholder(tf.float32, shape=[None, targ_dim])
con_v = tf.placeholder(tf.float32, shape=[None, con_dim])

# variables of discriminator
D_W1 = tf.Variable(xavier_init([targ_dim + con_dim, h_dim1]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim1]))
D_W2 = tf.Variable(xavier_init([h_dim1, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))
theta_D = [D_W1, D_W2, D_b1, D_b2]

# placeholder of generator
Noise = tf.placeholder(tf.float32, shape=[None, Noise_dim])

# variables of generator
G_W1 = tf.Variable(xavier_init([Noise_dim + con_dim, h_dim1]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim1]))
G_W2 = tf.Variable(xavier_init([h_dim1, targ_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[targ_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]

def discriminator(x, y):
    '''
    The discriminator for GANs
    Inputs: 
        x -the training data or generated data
        y -the radias
    '''
    inputs = tf.concat(axis=1, values=[x, y])
    D_h1 = tf.layers.batch_normalization((tf.matmul(inputs, D_W1) + D_b1), True)
    D_h1 = lrelu(D_h1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit

# generator
def generator(Noise, y):
    '''
    The generator for GANs
    Inputs: 
        Noise -the latent variables
    '''
    inputs = tf.concat(axis=1, values=[Noise, y])
    G_h1 = tf.matmul(inputs, G_W1) + G_b1
    G_h1= lrelu(tf.layers.batch_normalization(G_h1, True))
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.tanh(G_log_prob)
    return G_prob

def constraints(x, y):
    '''
    The constraints for circle
    Inputs:
        x  - circle data
        y  - the radia of circle data
    Outputs:
        C_approx - the value of constraints
    '''
    # x: mb_size*(2*num_point)
    x = tf.reshape(x, [mb_size, 2, num_point])
    # x: mb_size*num_point
    x = tf.reduce_sum(tf.square(x), 1) 
    
    # change y^2 from mb_size*1 to mb_size*num_point
    y = tf.tile(tf.square(y), [1, num_point])
    
    # (x1^2+x2^2-y^2)^2
    C_approx = tf.square(x - y)
    kesi  = tf.ones(tf.shape(C_approx))*(cons_value**2)
    C_approx = C_approx-kesi
    C_approx = tf.reduce_sum(tf.nn.relu(C_approx), 1)
    return C_approx

G_sample = generator(Noise, con_v)
D_real = discriminator(target, con_v)
D_fake = discriminator(G_sample, con_v)

penalty = tf.reduce_mean(constraints(G_sample, con_v))
penalty_log = tf.log(penalty+1)

# WGAN
D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake) 
G_loss = -tf.reduce_mean(D_fake)+lam_constraint*tf.reduce_mean(penalty_log)

D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(-D_loss, var_list=theta_D))
G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(G_loss, var_list=theta_G))

clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

# samples for testing
n_pred = 5000
noise_pred = sample_Z(n_pred, Noise_dim)
circle_real, con_pred = generate_samples(num_samp=n_pred, num_point=num_point)

# training samples
num_samp = 5000
circle, label = generate_samples(num_samp=num_samp, num_point=num_point)

# save the record
record_epoch = []
record_loss_D = []
record_loss_G = []
record_deviation = []
record_bias = []

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print('Number of sample: {}'.format(5000))  
    for it in range(epoch+1):
        for iter in range(num_samp//mb_size):
            output_sample = circle[iter*mb_size:(iter+1)*mb_size]
            con_sample = label[iter*mb_size:(iter+1)*mb_size]
            noise_sample = sample_Z(mb_size, Noise_dim)
            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={target: output_sample, Noise: noise_sample, con_v:con_sample})
            _, G_loss_curr, penalty_curr = sess.run([G_solver, G_loss, penalty], feed_dict={target: output_sample, Noise: noise_sample, con_v:con_sample})
                   
        if it % (epoch/20) == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'.format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print('penalty: {:.4}'.format(penalty_curr))
            record_epoch.append(it)
            record_loss_D.append(D_loss_curr)
            record_loss_G.append(G_loss_curr)
            
            # prediction
            target_pred = sess.run(G_sample, feed_dict={Noise: noise_pred, con_v: con_pred})
            
            deviation = 0
            bias = 0
            for i, sample in enumerate(target_pred):
                data_pred = np.reshape(sample, (2, num_point))
                x_pred = data_pred[0]
                y_pred = data_pred[1]
                
                # fit
                popt_x, pcov_x = curve_fit(fun_fit, t, x_pred)
                popt_y, pcov_y = curve_fit(fun_fit, t, y_pred)
                
                # calculate r with fitting
                r = (np.abs(popt_x[0])+np.abs(popt_y[0]))/2
                deviation += np.sum((x_pred**2+y_pred**2-r**2)**2)/100
                bias += np.abs(con_pred[i]-r)
                
            record_bias.append(bias)
            record_deviation.append(deviation)
            name = 'soft_constraint_'+str(it)
            plot_cGANs_ring(target_pred, n, circle_real, con_pred, name=name)