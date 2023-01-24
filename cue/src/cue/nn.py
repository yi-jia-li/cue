### Speculator Neural Net
import numpy as np
import glob
import tensorflow as tf
import tqdm

import pickle
dtype = tf.float32

class Speculator(tf.keras.Model):
    """
    SPECULATOR model
    """

    def __init__(self, n_parameters=None, wavelengths=None, pca_transform_matrix=None, parameters_shift=None, parameters_scale=None, pca_shift=None, pca_scale=None, log_spectrum_shift=None, log_spectrum_scale=None, n_hidden=[50,50], restore=False, restore_filename=None, trainable=True, optimizer=tf.keras.optimizers.Adam()):
        
        """
        Constructor.
        :param n_parameters: number of SED model parameters (inputs to the network)
        :param n_wavelengths: number of wavelengths in the modelled SEDs
        :param pca_transform_matrix: the PCA basis vectors, ie., an [n_pcas x n_wavelengths] matrix
        :param parameters_shift: shift for input parameters
        :param parameters_scalet: scale for input parameters
        :param pca_shift: shift for PCA coefficients
        :param pca_scale: scale for PCA coefficients
        :param log_spectrum_shift: shift for the output spectra
        :param log_spectrum_scale: scale for the output spectra
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param restore: (bool) whether to restore an previously trained model or not
        :param restore_filename: filename tag (without suffix) for restoring trained model from file (this will be a pickle file with all of the model attributes and weights)
        """
        
        # super
        super(Speculator, self).__init__()
        
        # restore
        if restore is True:
            self.restore(restore_filename)
            
        # else set variables from input parameters
        else:
            # parameters
            self.n_parameters = n_parameters
            self.n_wavelengths = pca_transform_matrix.shape[-1]
            self.n_pcas = pca_transform_matrix.shape[0]
            self.n_hidden = n_hidden
            self.wavelengths = wavelengths

            # architecture
            self.architecture = [self.n_parameters] + self.n_hidden + [self.n_pcas]
            self.n_layers = len(self.architecture) - 1

            # PCA transform matrix
            self.pca_transform_matrix_ = pca_transform_matrix

            # shifts and scales...
        
            # input parameters shift and scale
            self.parameters_shift_ = parameters_shift if parameters_shift is not None else np.zeros(self.n_parameters)
            self.parameters_scale_ = parameters_scale if parameters_scale is not None else np.ones(self.n_parameters)

            # PCA shift and scale
            self.pca_shift_ = pca_shift if pca_shift is not None else np.zeros(self.n_pcas)
            self.pca_scale_ = pca_scale if pca_scale is not None else np.ones(self.n_pcas)

            # spectrum shift and scale
            self.log_spectrum_shift_ = log_spectrum_shift if log_spectrum_shift is not None else np.zeros(self.n_wavelengths)
            self.log_spectrum_scale_ = log_spectrum_scale if log_spectrum_scale is not None else np.ones(self.n_wavelengths)

        # shifts and scales and transform matrix into tensorflow constants...
        
        # input parameters shift and scale
        self.parameters_shift = tf.constant(self.parameters_shift_, dtype=dtype, name='parameters_shift')
        self.parameters_scale = tf.constant(self.parameters_scale_, dtype=dtype, name='parameters_scale')
        
        # PCA shift and scale
        self.pca_shift = tf.constant(self.pca_shift_, dtype=dtype, name='pca_shift')
        self.pca_scale = tf.constant(self.pca_scale_, dtype=dtype, name='pca_scale')
        
        # spectrum shift and scale
        self.log_spectrum_shift = tf.constant(self.log_spectrum_shift_, dtype=dtype, name='log_spectrum_shift')
        self.log_spectrum_scale = tf.constant(self.log_spectrum_scale_, dtype=dtype, name='log_spectrum_scale')
        
        # pca transform matrix
        self.pca_transform_matrix = tf.constant(self.pca_transform_matrix_, dtype=dtype, name='pca_transform_matrix')
        
        # trainable variables...
        
        # weights, biases and activation function parameters for each layer of the network
        self.W = []
        self.b = []
        self.alphas = []
        self.betas = [] 
        for i in range(self.n_layers):
            self.W.append(tf.Variable(tf.random.normal([self.architecture[i], self.architecture[i+1]], 0., np.sqrt(2./self.n_parameters)), name="W_" + str(i), trainable=trainable))
            self.b.append(tf.Variable(tf.zeros([self.architecture[i+1]]), name = "b_" + str(i), trainable=trainable))
        for i in range(self.n_layers-1):
            self.alphas.append(tf.Variable(tf.random.normal([self.architecture[i+1]]), name = "alphas_" + str(i), trainable=trainable))
            self.betas.append(tf.Variable(tf.random.normal([self.architecture[i+1]]), name = "betas_" + str(i), trainable=trainable))
        
        # restore weights if restore = True
        if restore is True:
            for i in range(self.n_layers):
                self.W[i].assign(self.W_[i])
                self.b[i].assign(self.b_[i])
            for i in range(self.n_layers-1):
                self.alphas[i].assign(self.alphas_[i])
                self.betas[i].assign(self.betas_[i])

        self.optimizer = optimizer
            
    # non-linear activation function
    def activation(self, x, alpha, beta):
        
        return tf.multiply(tf.add(beta, tf.multiply(tf.sigmoid(tf.multiply(alpha, x)), tf.subtract(1.0, beta)) ), x)

    # call: forward pass through the network to predict pca coefficients
    @tf.function
    def call(self, parameters):
        
        outputs = []
        layers = [tf.divide(tf.subtract(parameters, self.parameters_shift), self.parameters_scale)]
        for i in range(self.n_layers - 1):
            
            # linear network operation
            outputs.append(tf.add(tf.matmul(layers[-1], self.W[i]), self.b[i]))
            
            # non-linear activation function
            layers.append(self.activation(outputs[-1], self.alphas[i], self.betas[i]))

        # linear output layer
        layers.append(tf.add(tf.matmul(layers[-1], self.W[-1]), self.b[-1]))
            
        # rescale the output (predicted PCA coefficients) and return
        return tf.add(tf.multiply(layers[-1], self.pca_scale), self.pca_shift)
            
    # pass inputs through the network to predict spectrum
    @tf.function
    def log_spectrum(self, parameters):
        
        # pass through network to compute PCA coefficients
        pca_coefficients = self.call(parameters)
        
        # transform from PCA to normalized spectrum basis; shift and re-scale normalized spectrum -> spectrum
        return tf.add(tf.multiply(tf.matmul(pca_coefficients, self.pca_transform_matrix), self.log_spectrum_scale), self.log_spectrum_shift)
    
    # save network parameters to numpy arrays
    def update_emulator_parameters(self):
        
        # put network parameters to numpy arrays
        self.W_ = [self.W[i].numpy() for i in range(self.n_layers)]
        self.b_ = [self.b[i].numpy() for i in range(self.n_layers)]
        self.alphas_ = [self.alphas[i].numpy() for i in range(self.n_layers-1)]
        self.betas_ = [self.betas[i].numpy() for i in range(self.n_layers-1)]
        
        # put shift and scale parameters to numpy arrays
        self.parameters_shift_ = self.parameters_shift.numpy()
        self.parameters_scale_ = self.parameters_scale.numpy()
        self.pca_shift_ = self.pca_shift.numpy()
        self.pca_scale_ = self.pca_scale.numpy()
        self.log_spectrum_shift_ = self.log_spectrum_shift.numpy()
        self.log_spectrum_scale_ = self.log_spectrum_scale.numpy()
        
        # pca transform matrix
        self.pca_transform_matrix_ = self.pca_transform_matrix.numpy()
        
    # save
    def save(self, filename):
 
        # attributes
        attributes = [self.W_, 
                      self.b_, 
                      self.alphas_, 
                      self.betas_, 
                      self.parameters_shift_, 
                      self.parameters_scale_,
                      self.pca_shift_,
                      self.pca_scale_,
                      self.log_spectrum_shift_,
                      self.log_spectrum_scale_,
                      self.pca_transform_matrix_,
                      self.n_parameters,
                      self.n_wavelengths,
                      self.wavelengths,
                      self.n_pcas,
                      self.n_hidden,
                      self.n_layers,
                      self.architecture]
        
        # save attributes to file
        f = open(filename + ".pkl", 'wb')
        pickle.dump(attributes, f)
        f.close()
        
    # restore attributes
    def restore(self, filename):
        
        # load attributes
        f = open(filename + ".pkl", 'rb')
        self.W_, self.b_, self.alphas_, self.betas_, self.parameters_shift_, \
        self.parameters_scale_, self.pca_shift_, self.pca_scale_, self.log_spectrum_shift_, \
        self.log_spectrum_scale_, self.pca_transform_matrix_, self.n_parameters, self.n_wavelengths, \
        self.wavelengths, self.n_pcas, self.n_hidden, self.n_layers, self.architecture = pickle.load(f)
        f.close()
    
    # forward prediction of spectrum given input parameters implemented in numpy
    def log_spectrum_(self, parameters):
        
        # forward pass through the network
        act = []
        layers = [(parameters - self.parameters_shift_)/self.parameters_scale_]
        for i in range(self.n_layers-1):

            # linear network operation
            act.append(np.dot(layers[-1], self.W_[i]) + self.b_[i])

            # pass through activation function
            layers.append((self.betas_[i] + (1.-self.betas_[i])*1./(1.+np.exp(-self.alphas_[i]*act[-1])))*act[-1])

        # final (linear) layer -> (normalized) PCA coefficients
        layers.append(np.dot(layers[-1], self.W_[-1]) + self.b_[-1])

        # rescale PCA coefficients, multiply out PCA basis -> normalized spectrum, shift and re-scale spectrum -> output spectrum
        return layers[-1]*self.pca_scale_ + self.pca_shift_ #np.dot(layers[-1]*self.pca_scale_ + self.pca_shift_, self.pca_transform_matrix_)*self.log_spectrum_scale_ + self.log_spectrum_shift_

    ### Infrastructure for network training ###

    @tf.function
    def compute_loss_spectra(self, spectra, parameters, noise_floor):

      return tf.sqrt(tf.reduce_mean(tf.divide(tf.math.squared_difference(tf.exp(self.log_spectrum(parameters)), spectra), tf.square(noise_floor))))      

    @tf.function
    def compute_loss_and_gradients_spectra(self, spectra, parameters, noise_floor):

      # compute loss on the tape
      with tf.GradientTape() as tape:

        # loss
        loss = tf.sqrt(tf.reduce_mean(tf.divide(tf.math.squared_difference(tf.exp(self.log_spectrum(parameters)), spectra), tf.square(noise_floor)))) 

      # compute gradients
      gradients = tape.gradient(loss, self.trainable_variables)

      return loss, gradients

    def training_step_spectra(self, spectra, parameters, noise_floor):

      # compute loss and gradients
      loss, gradients = self.compute_loss_and_gradients_spectra(spectra, parameters, noise_floor)

      # apply gradients
      self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

      return loss

    def training_step_with_accumulated_gradients_spectra(self, spectra, parameters, noise_floor, accumulation_steps=10):

      # create dataset to do sub-calculations over
      dataset = tf.data.Dataset.from_tensor_slices((spectra, parameters, noise_floor)).batch(int(spectra.shape[0]/accumulation_steps))

      # initialize gradients and loss (to zero)
      accumulated_gradients = [tf.Variable(tf.zeros_like(variable), trainable=False) for variable in self.trainable_variables]
      accumulated_loss = tf.Variable(0., trainable=False)

      # loop over sub-batches
      for spectra_, parameters_, noise_floor_ in dataset:
        
        # calculate loss and gradients
        loss, gradients = self.compute_loss_and_gradients_spectra(spectra_, parameters_, noise_floor_)

        # update the accumulated gradients and loss
        for i in range(len(accumulated_gradients)):
          accumulated_gradients[i].assign_add(gradients[i]*spectra_.shape[0]/spectra.shape[0])
        accumulated_loss.assign_add(loss*spectra_.shape[0]/spectra.shape[0])
        
        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(accumulated_gradients, self.trainable_variables))

      return accumulated_loss

    @tf.function
    def compute_loss_pca(self, pca, parameters):

      return tf.sqrt(tf.reduce_mean(tf.math.squared_difference(self.call(parameters), pca)))      

    @tf.function
    def compute_loss_and_gradients_pca(self, pca, parameters):

      # compute loss on the tape
      with tf.GradientTape() as tape:

        # loss
        loss = tf.sqrt(tf.reduce_mean(tf.math.squared_difference(self.call(parameters), pca))) 

      # compute gradients
      gradients = tape.gradient(loss, self.trainable_variables)

      return loss, gradients

    def training_step_pca(self, pca, parameters):

      # compute loss and gradients
      loss, gradients = self.compute_loss_and_gradients_pca(pca, parameters)

      # apply gradients
      self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

      return loss

    def training_step_with_accumulated_gradients_pca(self, pca, parameters, accumulation_steps=10):

      # create dataset to do sub-calculations over
      dataset = tf.data.Dataset.from_tensor_slices((pca, parameters)).batch(int(pca.shape[0]/accumulation_steps))

      # initialize gradients and loss (to zero)
      accumulated_gradients = [tf.Variable(tf.zeros_like(variable), trainable=False) for variable in self.trainable_variables]
      accumulated_loss = tf.Variable(0., trainable=False)

      # loop over sub-batches
      for pca_, parameters_ in dataset:
        
        # calculate loss and gradients
        loss, gradients = self.compute_loss_and_gradients_pca(pca_, parameters_)

        # update the accumulated gradients and loss
        for i in range(len(accumulated_gradients)):
          accumulated_gradients[i].assign_add(gradients[i]*pca_.shape[0]/pca.shape[0])
        accumulated_loss.assign_add(loss*pca_.shape[0]/pca.shape[0])
        
        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(accumulated_gradients, self.trainable_variables))

      return accumulated_loss

    @tf.function
    def compute_loss_log_spectra(self, log_spectra, parameters):

      return tf.sqrt(tf.reduce_mean(tf.math.squared_difference(self.log_spectrum(parameters), log_spectra)))      

    @tf.function
    def compute_loss_and_gradients_log_spectra(self, log_spectra, parameters):

      # compute loss on the tape
      with tf.GradientTape() as tape:

        # loss
        loss = tf.sqrt(tf.reduce_mean(tf.math.squared_difference(self.log_spectrum(parameters), log_spectra))) 

      # compute gradients
      gradients = tape.gradient(loss, self.trainable_variables)

      return loss, gradients

    def training_step_log_spectra(self, log_spectra, parameters):

      # compute loss and gradients
      loss, gradients = self.compute_loss_and_gradients_log_spectra(log_spectra, parameters)

      # apply gradients
      self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

      return loss

    def training_step_with_accumulated_gradients_log_spectra(self, log_spectra, parameters, accumulation_steps=10):

      # create dataset to do sub-calculations over
      dataset = tf.data.Dataset.from_tensor_slices((log_spectra, parameters)).batch(int(log_spectra.shape[0]/accumulation_steps))

      # initialize gradients and loss (to zero)
      accumulated_gradients = [tf.Variable(tf.zeros_like(variable), trainable=False) for variable in self.trainable_variables]
      accumulated_loss = tf.Variable(0., trainable=False)

      # loop over sub-batches
      for log_spectra_, parameters_ in dataset:
        
        # calculate loss and gradients
        loss, gradients = self.compute_loss_and_gradients_log_spectra(log_spectra_, parameters_)

        # update the accumulated gradients and loss
        for i in range(len(accumulated_gradients)):
          accumulated_gradients[i].assign_add(gradients[i]*log_spectra_.shape[0]/log_spectra.shape[0])
        accumulated_loss.assign_add(loss*log_spectra_.shape[0]/log_spectra.shape[0])
        
        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(accumulated_gradients, self.trainable_variables))

      return accumulated_loss
