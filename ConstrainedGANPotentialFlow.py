#!/usr/bin/env python3
"""
Constrained GANs for potential flow
@author: zengyang
"""
# standard library imports
import os, time, math

# third party imports
import tensorflow as tf
import numpy as np

# the path for saving results
root = './PotentialflowResults/'
if not os.path.isdir(root):
    os.mkdir(root)

tf.reset_default_graph()
np.random.seed(1)

## Traing seting
# soft constrained value
epsilon = 0
# weight of constrained term
lam = 0.2
# training epoch
epoches = 20
# learning rate
learning_rate = 0.0005

## setting for potential flow
# mesh
mesh_size = 32
# channel of inputs
inputs_channel = 2
# batch size
batch_size = 100

print('epsilon: %.3f lambda: %.3f lr: %.6f ep: %.3f' %(epsilon, lam, learning_rate, epoches))

## functions
# generate samples
def GenerateSamples(mesh_size, parameters):
    ''' 
    generate the training samples
    two kinds of potential flows are used : Uniform and source
    Uniform: F1(z) = V*exp(-i*alpha)*z
    source:  F2(z) = m/(2*pi)*log(z)
    Inputs:
        n          -number size of mesh
        parameter  - V, alpha, m
    output:
        u, v the velocity of x and y direction
    '''
    # mesh
    x = [-0.5, 0.5]
    y = [-0.5, 0.5]
    x_mesh = np.linspace(x[0], x[1], int(mesh_size))
    y_mesh = np.linspace(y[0], y[1], int(mesh_size))
    X, Y = np.meshgrid(x_mesh, y_mesh)  
    U = []
    for i, p in enumerate(parameters):
        V = p[0]
        alpha  = p[1]
        m = p[2]
        
        # velocity of uniform
        u1 = np.ones([mesh_size, mesh_size])*V*np.cos(alpha)
        v1 = np.ones([mesh_size, mesh_size])*V*np.sin(alpha)
        
        # velocity of source
        u2 = m/(2*np.pi)*X/(X**2+Y**2)
        v2 = m/(2*np.pi)*Y/(X**2+Y**2)
        
        u = u1+u2
        v = v1+v2
        U_data = np.zeros([mesh_size, mesh_size, 2])
        U_data[:, :, 0] = u
        U_data[:, :, 1] = v
        U.append(U_data)
    return X, Y, np.asarray(U)

def NextBatch(num, labels, U):
    '''
    Return a total of `num` random samples and labels. 
    Inputs: 
        num    -number of output samples
        labels -the labels of samples
        circle -samples
    Output:
        np.asarray(U_shuffle)      -random samples
        np.asarray(label_shuffle)  -random labels
    '''
    idx = np.arange(0 , len(labels))
    np.random.shuffle(idx)
    idx = idx[:num]
    U_shuffle = [U[i] for i in idx]
    label_shuffle = [labels[i] for i in idx]
    return np.asarray(U_shuffle), np.asarray(label_shuffle)
    
# leak_relu
def LReLu(X, leak=0.2):
    '''
    LReLu activate function for Generator and Discriminator
    '''
    f1 = 0.5*(1+leak)
    f2 = 0.5*(1+leak)
    return f1*X+f2*tf.abs(X)


def Generator(z, isTrain=True, reuse=False):
    '''
    Generator of GANs
    '''
    with tf.variable_scope('Generator', reuse=reuse):
        # initializer
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        b_init = tf.constant_initializer(0.0)
        
        deconv1 = tf.layers.conv2d_transpose(z, 64, [4, 4], strides=(1, 1), padding='valid', 
                                             kernel_initializer=w_init, bias_initializer=b_init)
        LReLu1 = LReLu(tf.layers.batch_normalization(deconv1, training=isTrain), 0.2)
        # 2nd hidden layer
        deconv2 = tf.layers.conv2d_transpose(LReLu1, 64, [5, 5], strides=(2, 2), padding='same', 
                                             kernel_initializer=w_init, bias_initializer=b_init)
        LReLu2 = LReLu(tf.layers.batch_normalization(deconv2, training=isTrain), 0.2)
        
        # 3rd layer
        deconv3 = tf.layers.conv2d_transpose(LReLu2, 64, [5, 5], strides=(2, 2), padding='same', 
                                             kernel_initializer=w_init, bias_initializer=b_init)
        LReLu3 = LReLu(tf.layers.batch_normalization(deconv3, training=isTrain), 0.2)
        
        # output layer
        deconv4 = tf.layers.conv2d_transpose(LReLu3, 2, [5, 5], strides=(2, 2), padding='same', 
                                             kernel_initializer=w_init, bias_initializer=b_init)
        o = tf.nn.tanh(deconv4)
        return o

# D(x)
def Discriminator(x, isTrain=True, reuse=False):
    '''
    Discriminator of GANs
    '''
    with tf.variable_scope('Discriminator', reuse=reuse):
        # initializer
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        b_init = tf.constant_initializer(0.0)

        # 1st hidden layer
        conv1 = tf.layers.conv2d(x, 64, [5, 5], strides=(2, 2), padding='same', 
                                 kernel_initializer=w_init, bias_initializer=b_init)
        LReLu1 = LReLu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d(LReLu1, 64, [5, 5], strides=(2, 2), padding='same', 
                                 kernel_initializer=w_init, bias_initializer=b_init)
        LReLu2 = LReLu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)
        
        # 3rd hidden layer
        conv3 = tf.layers.conv2d(LReLu2, 64, [5, 5], strides=(2, 2), padding='same', 
                                 kernel_initializer=w_init, bias_initializer=b_init)
        LReLu3 = LReLu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        # output layer
        conv4 = tf.layers.conv2d(LReLu3, 1, [4, 4], strides=(1, 1), padding='valid', 
                                 kernel_initializer=w_init)
        o = tf.nn.sigmoid(conv4)

        return o, conv4

def Constraints(U, dx, dy, filtertf):
    '''
    This function is the constraints of potentianl flow: Div(U) = 0
    Inputs:
        U            -generated velocity
        dx           -differential of x
        dy           -differential of y
        filtertf     -to filter the singular point of divergence
    Outputs:
        penalty_loss    -value of constraints
        divergence_mean -mean of divergence
    '''
    # inverse normalization
    U = U*(1.1*(nor_max_v-nor_min_v)/2)+(nor_max_v+nor_min_v)/2
    # x.shape [batch_size, mesh_size, mesh_size, 2]
    u = tf.slice(U, [0,0,0,0], [batch_size, mesh_size, mesh_size, 1])
    v = tf.slice(U, [0,0,0,1], [batch_size, mesh_size, mesh_size, 1])
    
    u = tf.reshape(u, [batch_size, mesh_size, mesh_size])
    v = tf.reshape(v, [batch_size, mesh_size, mesh_size])
    
    u_left  = tf.slice(u, [0,0,0], [batch_size, mesh_size, mesh_size-1])
    u_right = tf.slice(u, [0,0,1], [batch_size, mesh_size, mesh_size-1])
    d_u     = tf.divide(tf.subtract(u_right, u_left), dx)
    
    v_up   = tf.slice(v, [0,0,0], [batch_size, mesh_size-1, mesh_size])
    v_down = tf.slice(v, [0,1,0], [batch_size, mesh_size-1, mesh_size])
    d_v    = tf.divide(tf.subtract(v_down, v_up), dy)
    
    delta_u = tf.slice(d_u, [0,1,0],[batch_size, mesh_size-1, mesh_size-1])
    delta_v = tf.slice(d_v, [0,0,1],[batch_size, mesh_size-1, mesh_size-1])
    
    divergence_field = delta_u+delta_v
    #filter divergence
    divergence_filter = tf.multiply(divergence_field, filtertf)
    divergence_square = tf.square(divergence_filter)
    divergence_mean = tf.reduce_mean(tf.reduce_mean(divergence_square,2), 1)
    
    # soft Constraints
    kesi = tf.ones(tf.shape(divergence_mean))*(epsilon)
    # keep abs(Div-epsilon)>0
    penalty_loss = tf.nn.relu(divergence_mean - kesi)
    return penalty_loss, divergence_mean

## generated training samples
# setting of training samples
n_sam = 2000
V_mu, V_sigma = 4, 0.8
alpha_mu, alpha_sigma = 0, np.pi/4
m_mu, m_sigma = 1, 0.2
V_sample     = np.random.normal(V_mu, V_sigma, n_sam)
alpha_sample = np.random.normal(alpha_mu, alpha_sigma, n_sam)
m_sample     = np.random.normal(m_mu, m_sigma, n_sam)
samples = np.zeros([n_sam, 3])
samples[:,0] = V_sample
samples[:,1] = alpha_sample
samples[:,2] = m_sample

#training samples
X, Y, U = GenerateSamples(mesh_size, parameters=samples)

# normalization
nor_max_v = np.max(U)
nor_min_v = np.min(U)

# compress the samples into [-1, 1]
U_training = (U-(nor_max_v+nor_min_v)/2)/(1.1*(nor_max_v-nor_min_v)/2)

# use to calculate divergence
d_x = X[:,1:]-X[:,:-1]
d_y = Y[1:,:]-Y[:-1,:]
d_x_batch = np.tile(d_x, (batch_size, 1)).reshape([batch_size, mesh_size, mesh_size-1])
d_y_batch = np.tile(d_y, (batch_size, 1)).reshape([batch_size, mesh_size-1, mesh_size])

# use to filter divergence
filter_div = np.ones((mesh_size-1, mesh_size-1))
filter_div[13:18,13:18] = 0
filter_batch = np.tile(filter_div, (batch_size, 1)).reshape([batch_size, mesh_size-1, mesh_size-1])

global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(learning_rate, global_step, 500, 0.95, staircase=True)

# placeholder
x = tf.placeholder(tf.float32, shape=(None, mesh_size, mesh_size, inputs_channel))
z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))
isTrain = tf.placeholder(dtype=tf.bool)

dx = tf.placeholder(tf.float32, shape=(None, mesh_size, mesh_size-1))
dy = tf.placeholder(tf.float32, shape=(None, mesh_size-1, mesh_size))
filtertf = tf.placeholder(tf.float32, shape=(None, mesh_size-1, mesh_size-1))

# networks : Generator
G_z = Generator(z, isTrain)

# networks : Discriminator
D_real, D_real_logits = Discriminator(x, isTrain)
D_fake, D_fake_logits = Discriminator(G_z, isTrain, reuse=tf.AUTO_REUSE)
penalty_loss, divergence_mean = Constraints(G_z, dx, dy, filtertf)

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('Discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('Generator')]

lam_GP = 10

# WGAN-GP
eps = tf.random_uniform([batch_size, 1], minval=0., maxval=1.)
eps = tf.reshape(eps,[batch_size, 1, 1, 1])
eps = eps * np.ones([batch_size, mesh_size, mesh_size, inputs_channel])
X_inter = eps*x + (1. - eps)*G_z
grad = tf.gradients(Discriminator(X_inter, isTrain, reuse=tf.AUTO_REUSE), [X_inter])[0]
grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
grad_pen = lam_GP * tf.reduce_mean((grad_norm - 1)**2)

# loss for each network
D_loss_real = -tf.reduce_mean(D_real_logits)
D_loss_fake = tf.reduce_mean(D_fake_logits)
D_loss = D_loss_real + D_loss_fake + grad_pen
delta_loss = tf.reduce_mean(penalty_loss)
G_loss_only = -tf.reduce_mean(D_fake_logits)
G_loss = G_loss_only + lam*tf.log(delta_loss+1)

# optimizer for each network 
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    optim = tf.train.AdamOptimizer(lr, beta1=0.5)
    D_optim = optim.minimize(D_loss, global_step=global_step, var_list=D_vars)
    G_optim = optim.minimize(G_loss, var_list=G_vars)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# save the record
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['delta_real'] = []
train_hist['penalty_loss'] = []
train_hist['generateddata'] = []
train_hist['generateddata_fit'] = []
train_hist['ratio'] = []

# save model and all variables
saver = tf.train.Saver()

# training-loop
print('training start!')
start_time = time.time()

for epoch in range(epoches+1):
    G_losses = []
    D_losses = []
    constraint_real_record = []
    penalty_loss_record = []
    epoch_start_time = time.time()
    for iter in range(U_training.shape[0] // batch_size):
        # training Discriminator
        x_ = U_training[iter*batch_size:(iter+1)*batch_size]
        z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))
        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, isTrain: True})
        
        # training Generator
        z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))
        loss_g_, _ = sess.run([G_loss, G_optim], {z:z_, x:x_, dx:d_x_batch, 
                              dy:d_y_batch, filtertf:filter_batch, isTrain: True})
        
        # monitor the training
        errD = D_loss.eval({z:z_, x:x_, filtertf:filter_batch, isTrain: False})
        errG = G_loss_only.eval({z: z_, dx:d_x_batch, dy:d_y_batch, filtertf:filter_batch, 
                                 isTrain: False})
        errdelta_real   = divergence_mean.eval({z:z_, dx:d_x_batch, dy:d_y_batch, 
                                                filtertf:filter_batch, isTrain: False})
        errpenalty_loss = penalty_loss.eval({z:z_, dx:d_x_batch, dy:d_y_batch, 
                                             filtertf:filter_batch, isTrain: False})
        
        D_losses.append(errD)
        G_losses.append(errG)
        constraint_real_record.append(errdelta_real)
        penalty_loss_record.append(errpenalty_loss)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f, Divergence: %.3f' % 
          ((epoch + 1), epoches, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses), 
           np.mean(constraint_real_record)))
    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['delta_real'].append(np.mean(constraint_real_record))
    train_hist['penalty_loss'].append(np.mean(penalty_loss_record))
    
    # generate velocity
    z_gene = np.random.normal(0, 1, (16, 1, 1, 100))
    generateddata = G_z.eval({z:z_gene, isTrain: False})
    generateddata = generateddata*(1.1*(nor_max_v-nor_min_v)/2)+(nor_max_v+nor_min_v)/2   
    train_hist['generateddata'].append(generateddata)
    if epoch % 20 == 0:
        z_gene = np.random.normal(0, 1, (2000, 1, 1, 100))
        generateddata = G_z.eval({z:z_gene, isTrain: False})
        generateddata = generateddata*(1.1*(nor_max_v-nor_min_v)/2)+(nor_max_v+nor_min_v)/2
        train_hist['generateddata_fit'].append(generateddata)

end_time = time.time()
total_ptime = end_time - start_time
name_data = root + 'ConsGANsPF-epsilon'+str(epsilon)+'-lam'+str(lam)+'-lr'+str(learning_rate)+'-ep'+str(epoches)
np.savez_compressed(name_data, a=train_hist, b=per_epoch_ptime)
save_model = name_data+'.ckpt'
save_path = saver.save(sess, save_model)