from prepare import *
from model_utils import *
from config import *

import os
import sys
import random
import time
import scipy.misc as misc
import sklearn.metrics as metrics
import tensorflow as tf
import numpy as np
from shutil import copyfile
import pdb
from time import gmtime, strftime
from sys import argv
import hickle as hkl
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class GENGAN:

    def __init__(self, save_name, load_name, patch_size, num_iterations,
                 batch_size, new_model, sample_rates, ckpt_num, train_vgg, load_vgg, load_weights, limits):
        self.sess = None
        self.save_name = save_name
        self.patch_size = patch_size
        self.load_name = load_name
        self.learn_rate = None
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.new_model = new_model
        self.sample_rates = sample_rates
        self.use_c = True
        self.ckpt_num = ckpt_num
        self.train_vgg = train_vgg
        self.load_vgg = load_vgg
        self.load_weights = load_weights
        self.limits = limits
        self.is_vanilla = False

        self.input_x = self.input_mask = self.input_real = self.input_boundary = None
        self.input_c = self.input_z = self.fake_image = self.global_step = None
        self.t_vars = self.d_vars = self.g_vars = self.c_vars = None
        self.D_real_attr = self.discriminator = None
        self.saver = self.d_saver = self.g_saver = None
        self.G_loss = self.D_loss = None
        self.G_solver = self.D_solver = self.G_loss_vgg = None
        self.VGG_solver = self.boundary_loss = self.boundary_solver = self.L1_loss = self.L1_solver = None

        # L1 and boundary loss params
        self.alpha = 0.7
        self.l1_factor = 500.0
        self.boundary_factor = 1000.0

    def build_model(self):
        print 'Building model'

        # Learning rate params
        self.global_step = tf.Variable(0, trainable=False)
        self.learn_rate = tf.train.exponential_decay(learn_rate, self.global_step,
                                                     500, 0.99, staircase=False)

        # Input variables
        self.input_x = tf.placeholder(tf.float32, [None, self.patch_size, self.patch_size, 1])
        self.input_mask = tf.placeholder(tf.float32, [None, self.patch_size, self.patch_size, 1])
        self.input_real = tf.placeholder(tf.float32, [None, self.patch_size, self.patch_size, 1])
        self.input_boundary = tf.placeholder(tf.float32, [None, self.patch_size, self.patch_size, 1])
        self.input_c = tf.placeholder(tf.float32, [None, 1, 1, c_dims])
        self.input_z = tf.placeholder(tf.float32, [None, 1, 1, z_dims])
        self.fake_image = build_generator(self.input_x, self.input_mask, self.input_c, use_c=self.use_c)

        # Build discriminator
        self.discriminator = build_discriminator(input_x=self.input_x, input_c=self.input_c, use_c=self.use_c)
        D_real, self.D_real_attr = self.discriminator([self.input_real, self.input_c])
        D_fake, _ = self.discriminator([self.fake_image, self.input_c])

        # Set training variables
        self.t_vars = tf.trainable_variables()
        self.d_vars = [ var for var in self.t_vars if var.name.startswith('d_')]
        self.g_vars = [ var for var in self.t_vars if var.name.startswith('g_')]
        self.c_vars = [ var for var in self.t_vars if var not in self.d_vars and var not in self.g_vars]

        # Set savers
        self.saver = tf.train.Saver(self.g_vars + self.d_vars, max_to_keep=10000)
        self.g_saver = tf.train.Saver(self.g_vars)
        self.d_saver = tf.train.Saver(self.d_vars)

        # Set loss functions
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_real)))
        self.D_loss = (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))) / 2.0

        # Set solvers
        self.G_solver = tf.train.AdamOptimizer(learning_rate=self.learn_rate, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.G_loss, var_list=self.g_vars, global_step=self.global_step)
        self.D_solver = tf.train.AdamOptimizer(learning_rate=self.learn_rate, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.D_loss, var_list=self.d_vars, global_step=self.global_step)

        # Build VGG networks
        vgg_real_c = build_vgg19(tf.multiply(self.input_real, 1 - self.input_mask))
        vgg_fake_c = build_vgg19(tf.multiply(self.fake_image, 1 - self.input_mask), reuse=True)

        # Extract VGG weights
        self.G_loss_vgg = tf.reduce_mean(tf.abs(vgg_real_c['input'] - vgg_fake_c['input']))
        vgg_real = build_vgg19(tf.multiply(self.input_real, self.input_mask))
        vgg_fake = build_vgg19(tf.multiply(self.fake_image, self.input_mask), reuse=True)
        self.G_loss_vgg += tf.reduce_mean(tf.abs(vgg_real['input'] - vgg_fake['input']))
        for i in range(1, 4):
            conv_str = 'pool' + str(i)
            self.G_loss_vgg += tf.reduce_mean(tf.abs(vgg_real[conv_str] - vgg_fake[conv_str]))

        vgg_real = build_vgg19(tf.multiply(self.input_real, self.input_boundary))
        vgg_fake = build_vgg19(tf.multiply(self.fake_image, self.input_boundary), reuse=True)
        self.G_loss_vgg += tf.reduce_mean(tf.abs(vgg_real['input'] - vgg_fake['input']))
        for i in range(1, 4):
            conv_str = 'pool' + str(i)
            self.G_loss_vgg += tf.reduce_mean(tf.abs(vgg_real[conv_str] - vgg_fake[conv_str]))

        # Set VGG solver
        self.VGG_solver = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.G_loss_vgg, global_step=self.global_step, var_list=self.g_vars)

        # L1 and boundary loss
        self.L1_loss = self.l1_factor * \
                       tf.reduce_mean(tf.abs(self.alpha *
                       tf.multiply(self.input_mask, self.fake_image - self.input_real)) +
                       tf.abs((1 - self.alpha) *
                       tf.multiply(1 - self.input_mask, self.fake_image - self.input_real)))

        self.L1_solver = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.L1_loss, global_step=self.global_step, var_list=self.g_vars)
        self.boundary_loss = self.boundary_factor * tf.reduce_mean(tf.multiply(self.input_boundary, tf.abs(self.fake_image - self.input_real)))
        self.boundary_solver = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.boundary_loss,
                                                                                              global_step=self.global_step,
                                                                                              var_list=self.g_vars)

    def train_model(self):
        with tf.Session() as self.sess:
            self.sess.run(tf.global_variables_initializer())
            # If using existing model
            if not self.new_model:
                # Load the VGG loss trained model
                if self.load_vgg and tf.train.checkpoint_exists(models_dir + self.save_name + '_vgg'):
                    print 'Loading vgg'
                    self.g_saver.restore(self.sess, models_dir + self.save_name + '_vgg')
                # Load the GAN loss trained model
                elif self.load_name is not None and self.load_weights:
                    print 'Loading model', self.load_name
                    if self.ckpt_num is not None:
                        path = checkpoints_dir + self.load_name+'_'+self.ckpt_num
                    else:
                        path = models_dir + self.load_name
                    self.g_saver.restore(self.sess, path)
                    self.d_saver.restore(self.sess, path)

            # Create data generators
            # Generators generate 256x256px patches
            print 'Getting dataset'
            data_generator = generate_cpatches(self.batch_size, sample_rates=self.sample_rates,
                                               is_vanilla=self.is_vanilla, limits=self.limits, ctype=data_type)
            val_data_generator = generate_cpatches(1, sample_rates=self.sample_rates, is_vanilla=self.is_vanilla,
                                                   limits=['val_rand', 'val_rand'], is_val=True, ctype=data_type)

            # iteration number
            it = 0
            # number of VGG loss pre-training iterations
            vgg_iters = 100
            # Number of iterations per epoch for each type of loss
            d_iters = 5
            g_iters = 1
            boundary_iters = 10
            vgg_iters = 10
            max_iters = 5000
            # Alternatively, train the D or G until loss drops below threshold
            D_loss_threshold = 0.3
            G_loss_threshold = 0.3

            print 'Training model'
            # First train on VGG loss only
            if self.train_vgg:
                print 'Pre-training on VGG'
                for i in range(vgg_iters):
                    print 'Iteration', i
                    VGG_loss_cur = self.train(self.sess, self.VGG_solver, self.G_loss_vgg, generator=data_generator, iters=100)
                    print 'VGG loss', VGG_loss_cur
                    self.validate(i, val_data_generator, self.sess)
                    save(self.save_name + '_vgg', it, self.g_saver, self.sess)

            while it < int(self.num_iterations):

                it += 1
                for j in range(1):
                    D_timer = 0
                    D_loss_cur = 100.0
                    while D_loss_cur > D_loss_threshold and D_timer < max_iters:
                        D_loss_cur = self.train(self.sess, self.D_solver, self.D_loss, generator=data_generator, iters=d_iters)
                        print 'D_loss', D_loss_cur
                        it += d_iters
                        D_timer += 1

                    print '========'
                    G_timer = 0
                    G_loss_cur = 100.0
                    while G_loss_cur > G_loss_threshold and G_timer < max_iters:
                        G_loss_cur = self.train(self.sess, self.G_solver, self.G_loss, generator=data_generator, iters=g_iters)
                        print 'G loss', G_loss_cur
                        it += g_iters
                        G_timer += 1

                    print '========'

                # VGG LOSS

                for i in range(1):
                    VGG_loss_cur = self.train(self.sess, self.VGG_solver, self.G_loss_vgg, generator=data_generator, iters=vgg_iters)
                    it += vgg_iters
                    print 'VGG loss', VGG_loss_cur

                # BOUNDARY LOSS

                for i in range(1):
                    boundary_loss_cur = self.train(self.sess, self.boundary_solver, self.boundary_loss, generator=data_generator, iters=boundary_iters)
                    it += boundary_iters
                    print 'Boundary loss', boundary_loss_cur

                # Save model some of the time
                if np.random.random() > 0.25:
                    print 'Saving model'
                    ckpt = np.random.random() > 0.9
                    if ckpt:
                        print 'Saving checkpoint'
                    save(self.save_name, it, self.saver, self.sess, ckpt=ckpt)
                    self.validate(it, val_data_generator, self.sess)

            print 'Saving model'
            save(self.save_name, it, self.saver, self.sess)
        tf.reset_default_graph()

    def validate_model(self):
        with tf.Session() as self.sess:
            self.sess.run(tf.global_variables_initializer())
            self.saver.restore(self.sess, models_dir + self.save_name)
            data_generator_val = generate_cpatches(self.batch_size, sample_rates=[0.5, 0.5],
                                                   limits=['val_rand', 'val_rand'], ctype=data_type)
            # Save some random validation samples
            for i in range(1, 6):
                self.validate(i * 1000, data_generator_val, self.sess)

    def synthesize_dataset(self, num, batch_size=batch_size, limits=[None]*c_dims, sample_rates = [0.5, 0.5]):
        with tf.Session() as self.sess:
            # Load model
            self.sess.run(tf.global_variables_initializer())
            if self.ckpt_num is not None:
                self.g_saver.restore(self.sess, checkpoints_dir + self.load_name+'_'+str(self.ckpt_num))
                self.d_saver.restore(self.sess, checkpoints_dir + self.load_name+'_'+str(self.ckpt_num))
            else:
                self.g_saver.restore(self.sess, models_dir + self.load_name)
                self.d_saver.restore(self.sess, models_dir + self.load_name)

            # Create patch generator
            generator_syn = generate_cpatches(batch_size, sample_rates=sample_rates, limits=limits, ctype=data_type)

            X_train = np.zeros((num, batch_size, self.patch_size, self.patch_size, 1))
            y_train = np.zeros((num, batch_size, c_dims))
            for i in range(0, num):
                print 'Num ', i
                data_X, data_c = generator_syn.next()

                # Use normals to generate malignant, and vice versa
                data_c[:, :, :, 0] = 1-data_c[:, :, :, 0]
                data_c[:, :, :, 1] = 1-data_c[:, :, :, 1]
                data_x = data_X[:, :, :, 0:1]
                data_mask = data_X[:, :, :, 1:2]
                pred_img = self.sess.run(self.fake_image, feed_dict={
                    self.input_x: data_x,
                    self.input_mask: data_mask,
                    self.input_c: data_c
                   })
                X_train[i] = pred_img
                y_train[i] = data_c.reshape((-1, c_dims))

            X_train = X_train.reshape((-1, self.patch_size, self.patch_size, 1))
            y_train = y_train.reshape((-1, c_dims))
            return X_train, y_train

    def train(self, sess, solver, loss, generator, step=0, iters=10, return_acc=False):
        loss_avg = []
        accs = []
        for i in range(iters):
            data_X, data_c = generator.next()
            #pdb.set_trace()
            data_x = data_X[:, :, :, 0:1]
            data_mask = data_X[:, :, :, 1:2]
            data_real = data_X[:, :, :, 2:3]
            data_boundary = data_X[:, :, :, 3:4]
            #pdb.set_trace()
            data_z = np.random.random_sample(data_c.shape[0:3]+(z_dims,))
            #pdb.set_trace()
            _, loss_cur, attr = sess.run([solver, loss, self.D_real_attr],
               feed_dict={
               self.input_x: data_x,
               self.input_mask: data_mask,
               self.input_real: data_real,
               self.input_c: data_c,
               self.input_z: data_z,
               self.input_boundary: data_boundary
               })
            if return_acc:
                data_c = data_c.reshape((-1, c_dims))
                attr_s = one_hot(attr)
                accs.append(metrics.accuracy_score(data_c, attr_s))
            loss_avg.append(loss_cur)

        if return_acc:
            return (np.mean(loss_avg), np.mean(accs))
        else:
            return np.mean(loss_avg)

    def validate(self, i, data_generator, sess):
        print 'Validating', i
        data_X, data_c = data_generator.next()
        data_x = data_X[0:1, :, :, 0:1]
        data_mask = data_X[0:1, :, :, 1:2]
        data_real = data_X[0:1, :, :, 2:3]
        mask_image = data_mask[0, :, :, 0]
        real_image = data_real[0, :, :, 0]
        # Get non-cancer sample
        data_c = np.reshape(np.array([1, 0]), (1, 1, 1, c_dims))
        data_z = np.random.random_sample((1, 1, 1, z_dims))
        output_ben_mass = sess.run(self.fake_image, feed_dict={
            self.input_x: data_x,
            self.input_mask: data_mask,
            self.input_c: data_c,
            self.input_z: data_z
           })
        # Get cancer sample
        data_c = np.reshape(np.array([0, 1]), (1, 1, 1, c_dims))
        data_z = np.random.random_sample((1, 1, 1, z_dims))
        output_mal_mass = sess.run(self.fake_image, feed_dict={
            self.input_x: data_x,
            self.input_mask: data_mask,
            self.input_c: data_c,
            self.input_z: data_z
           })
        output_ben_mass = output_ben_mass[0, :, :, 0]
        output_mal_mass = output_mal_mass[0, :, :, 0]
        img = np.concatenate((real_image, mask_image, output_ben_mass, output_mal_mass), axis=1)
        img = scipy.ndimage.zoom(img, zoom=[0.75, 0.75])
        directory = './validation/' + self.save_name + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        print np.unique(output_ben_mass)
        print np.unique(output_mal_mass)
        scipy.misc.toimage(img, cmin=0.0, cmax=1.0).save(directory + self.save_name + '_' + str(self.patch_size) + '_' + str(i) + '.png')
