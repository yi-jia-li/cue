import jax
import jax.numpy as jnp
import numpy as np

def gelu(x):
    return jax.nn.gelu(x,approximate=True)

def linear(x):
    return x

def sigmoid(x):
    return jax.nn.sigmoid(x)

def speculator(x,alpha=None,beta=None):
    """
    Eq. 8 from Alsing et. al 2019 (Speculator paper). This is used for the `cue` emulator. 
    x = (xW + b) from previous layer.
    """
    # return (beta + (1.-beta)*1./(1.+jnp.exp(-alpha*x)))*x
    return (beta + (1. - beta) * jax.nn.sigmoid(alpha*x)) * x


available_activation_functions = {
    'gelu':gelu,
    'linear':linear,
    'sigmoid':sigmoid,
    'speculator':speculator
}

class ANN:
    """
    Framework for a fully-connected deep/artificial neural network. 
    """
    def __init__(self, params):
        """
        Initializes the Emulator with the dictionary of the ANN.

        Parameters:
        - params: Dictionary of the ANN parameters.
        """
        self.dict = params
        self.load_ann()

    def load_ann(self):
        """
        Loads the ANN data from the dictionary and sets up the emulator.
        """
        
        data = self.dict
        self.weights =  data['ann']['weights']
        self.biases = data['ann']['biases']
        self.norms = {key:jnp.array(val,dtype=jnp.float32) for key,val in data['normalizations'].items()}
        self.architecture = data['architecture']
        self._activation_functions = data['ann']['activation_functions']
        
        self.activation_functions = []
        for i,activation in enumerate(self._activation_functions):
            assert activation in list(available_activation_functions.keys()), f'The activation function {activation} is not an available activation function (options include {list(available_activation_functions.keys())}).'
            self.activation_functions.append(available_activation_functions[activation])
            
        if 'residuals' in data.keys():
            self.residuals = data['residuals']['varience']
        else:
            self.residuals = None

    def predict(self, theta):
        """        
        Makes a prediction using the emulator.

        Parameters:
        - theta: Input parameter array to predict on.

        Returns:
        - The predicted output in native output units of the emulator.
        """
        theta_norm = self.normalize(theta)
        prediction = self.forward_pass(theta_norm)
        result = self.denormalize(prediction)
        return result

    def normalize(self,x):
        """
        Normalizes the inputs into the network by the specified mean and standard deviation of the training set inputs.
        """
        return (x - self.norms['input_mu']) / self.norms['input_std']
    
    def denormalize(self,x):
        """
        Dermalizes the outputs of the network by the specified mean and standard deviation of the training set outputs.
        """
        return (x * self.norms['output_std']) + self.norms['output_mu']

    def forward_pass(self,x):
        """
        Performs a forward pass through the network.
    
        Args:
            params (list): List of tuples containing the weights and biases for each layer.
            x (array-like): Input data.
            activations (list): List of activation functions for each layer.
    
        Returns:
            array-like: Output of the network after the forward pass.
        """
    
        for i,activation in enumerate(self.activation_functions):
            x = activation(jnp.inner(x, self.weights[i]) + self.biases[i])
        return x


class SpeculatorANN(ANN):
    """
    This ANN subclass has a special activation function that follows eqn. 8 of Alsing et. al 2020. It has additional 
    parameters in the activation function that are trained, called alpha and beta in our case. In the context of `fspse`, 
    this is for the `cue` ANNs.
    """

    def __init__(self,*args):
        """
        Loads the additional ANN parameters from the dictionary and sets up the emulator.
        """
        super().__init__(*args)
        
        self.alphas = self.dict['ann']['alphas']
        self.betas = self.dict['ann']['betas']

    def forward_pass(self,x):
        """
        Performs a forward pass through the network with a modified speculator activation function.
    
        Args:
            params (list): List of tuples containing the weights and biases for each layer.
            x (array-like): Input data.
            activations (list): List of activation functions for each layer.
    
        Returns:
            array-like: Output of the network after the forward pass.
        """

        i = 0 # iterating through layers
        for activation, activation_name in zip(self.activation_functions, self._activation_functions):
            if activation_name == 'speculator': # this isn't quite right. It assumes that the alphas and betas nested lists are the same shape as the weights, or that the first layer always has this activation function. 
                x = activation(jnp.inner(x, self.weights[i]) + self.biases[i],alpha=self.alphas[i],beta=self.betas[i])
            else:
                x = activation(jnp.inner(x, self.weights[i]) + self.biases[i])
            i += 1
        return x
    
        
class PCABasis:
    """
    A JAX-compatible PCA Basis class for performing principal component analysis (PCA) 
    transformations and their inverses. This class uses pre-computed PCA components, 
    means, and variances to project data into PCA space and reconstruct it back to the 
    original space.

    Attributes:
        components (jnp.ndarray): The principal components matrix of shape (n_components, n_features).
        means (jnp.ndarray): The mean values for centering data, of shape (n_features,).
        variance (jnp.ndarray): The variance explained by each principal component, of shape (n_components,).
        scaled_components (jnp.ndarray): The scaled principal components where each component is scaled 
            by the square root of its variance.
    """

    def __init__(self, dict):
        """
        Initializes the PCA basis with pre-computed PCA components, means, and variances.

        Args:
            dict (dict): A dictionary containing the PCA components, means, and variances. 
                The keys should include:
                - 'components': Principal components matrix (n_components, n_features).
                - 'means': Mean values for each feature (n_features,).
                - 'variance': Variance explained by each principal component (n_components,).
        """
        self.components = dict['components']
        self.means = dict['means']
        # self.variance = dict['variance']
        # self.scaled_components = jnp.sqrt(self.variance)[:, jnp.newaxis] * self.components

    def transform(self, x):
        """
        Projects input data into PCA space.

        Args:
            x (jnp.ndarray): Input data of shape (n_samples, n_features) to be transformed. 
                Each row represents a data sample.

        Returns:
            jnp.ndarray: The transformed data of shape (n_samples, n_components).
        """
        x_centered = x - self.means
        x_transformed = jnp.dot(x_centered, self.components.T)
        return x_transformed

    def inverse_transform(self, x_transformed):
        """
        Reconstructs data from PCA space back to the original feature space.

        Args:
            x_transformed (jnp.ndarray): Data in PCA space of shape (n_samples, n_components), 
                typically the output of the `transform` method or a neural network.

        Returns:
            jnp.ndarray: The reconstructed data in the original feature space of shape (n_samples, n_features).
        """
        x = jnp.dot(x_transformed, self.components) + self.means
        return x
