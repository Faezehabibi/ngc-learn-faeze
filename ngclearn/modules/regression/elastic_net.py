from jax import random, jit
import numpy as np
from ngclearn.utils import weight_distribution as dist
from ngclearn import Context, numpy as jnp
from ngclearn.components import (RateCell,
                                 HebbianSynapse,
                                 GaussianErrorCell,
                                 StaticSynapse)
from ngclearn.utils.model_utils import scanner


class Iterative_ElasticNet():
    """
        A neural circuit implementation of the iterative Elastic Net (L1 and L2) algorithm
        using Hebbian learning update rule.

        The circuit implements sparse regression through Hebbian synapses with Elastic Net regularization.

        The specific differential equation that characterizes this model is dW_reg (for adjusting W, given
        dW (the gradient of loss/energy function), it adds lmbda * dW_reg to the dW)
        
        | dW_reg = (jnp.sign(W) * l1_ratio) + (W * (1-l1_ratio)/2)
        | dW/dt = dW + lmbda * dW_reg



        | --- Circuit Components: ---
        | W - HebbianSynapse for learning regularized dictionary weights
        | err - GaussianErrorCell for computing prediction errors
        | --- Component Compartments ---
        | W.inputs - input features (takes in external signals)
        | W.pre - pre-synaptic activity for Hebbian learning
        | W.post - post-synaptic error signals
        | W.weights - learned dictionary coefficients
        | err.mu - predicted outputs
        | err.target - target signals (target vector)
        | err.dmu - error gradients
        | err.L - loss/energy values

        Args:
            key: JAX PRNG key for random number generation

            name: string name for this solver

            sys_dim: dimensionality of the system/target space

            dict_dim: dimensionality of the dictionary/feature space/the number of predictors

            batch_size: number of samples to process in parallel

            weight_fill: initial constant value to fill weight matrix with (Default: 0.05)

            lr: learning rate for synaptic weight updates (Default: 0.01)

            lmbda: elastic net regularization lambda parameter (Default: 0.0001)

            optim_type: optimization type for updating weights; supported values are
                "sgd" and "adam" (Default: "adam")

            threshold: minimum absolute coefficient value - values below this are set
                to zero during thresholding (Default: 0.001)

            epochs: number of training epochs (Default: 100)
    """
    def __init__(self, key, name, sys_dim, dict_dim, batch_size, weight_fill=0.05, lr=0.01,
                 lmbda = 0.0001, l1_ratio=0.5, optim_type="adam", threshold=0.05, epochs=100):
        key, *subkeys = random.split(key, 10)

        ## synaptic plasticity properties and characteristics
        self.T = 100
        self.dt = 1
        self.epochs = epochs
        self.weight_fill = weight_fill
        self.threshold = threshold
        self.name = name
        self.lr = lr
        feature_dim = dict_dim

        with Context(self.name) as self.circuit:
            self.W = HebbianSynapse("W", shape=(feature_dim, sys_dim), eta=self.lr,
                                   sign_value=-1, weight_init=dist.constant(weight_fill),
                                   prior=('elastic_net', (lmbda, l1_ratio)), w_bound=0.,
                                   optim_type=optim_type, key=subkeys[0])
            self.err = GaussianErrorCell("err", n_units=sys_dim)

            # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.W.batch_size = batch_size
            self.err.batch_size = batch_size
            # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.err.mu << self.W.outputs
            self.W.post << self.err.dmu
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            advance_cmd, advance_args =self.circuit.compile_by_key(self.W,  ## execute prediction synapses
                                                               self.err,  ## finally, execute error neurons
                                                               compile_key="advance_state")
            evolve_cmd, evolve_args =self.circuit.compile_by_key(self.W, compile_key="evolve")
            reset_cmd, reset_args =self.circuit.compile_by_key(self.err, self.W, compile_key="reset")
            # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.dynamic()

    def dynamic(self):  ## create dynamic commands forself.circuit
        W, err = self.circuit.get_components("W", "err")
        self.self = W
        self.err = err

        @Context.dynamicCommand
        def batch_set(batch_size):
            self.W.batch_size = batch_size
            self.err.batch_size = batch_size

        @Context.dynamicCommand
        def clamps(y_scaled, X):
            self.W.inputs.set(X)
            self.W.pre.set(X)
            self.err.target.set(y_scaled)

        self.circuit.wrap_and_add_command(jit(self.circuit.evolve), name="evolve")
        self.circuit.wrap_and_add_command(jit(self.circuit.advance_state), name="advance")
        self.circuit.wrap_and_add_command(jit(self.circuit.reset), name="reset")


        @scanner
        def _process(compartment_values, args):
            _t, _dt = args
            compartment_values = self.circuit.advance_state(compartment_values, t=_t, dt=_dt)
            return compartment_values, compartment_values[self.W.weights.path]


    def thresholding(self, scale=1.):
        coef_old = self.coef_
        new_coeff = jnp.where(jnp.abs(coef_old) >= self.threshold, coef_old, 0.)

        self.coef_ = new_coeff * scale
        self.W.weights.set(new_coeff)

        return self.coef_, coef_old


    def fit(self, y, X):

        self.circuit.reset()
        self.circuit.clamps(y_scaled=y, X=X)

        for i in range(self.epochs):
            self.circuit._process(jnp.array([[self.dt * i, self.dt] for i in range(self.T)]))
            self.circuit.evolve(t=self.T, dt=self.dt)

        self.coef_ = np.array(self.W.weights.value)

        return self.coef_, self.err.mu.value, self.err.L.value





