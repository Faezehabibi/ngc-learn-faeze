from jax import random, numpy as jnp, jit
from ngclearn import resolver, Component, Compartment
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn.utils import tensorstats
from ngclearn.utils.weight_distribution import initialize_params
from ngcsimlib.logger import info

class DenseSynapse(JaxComponent): ## base dense synaptic cable
    """
    A dense synaptic cable; no form of synaptic evolution/adaptation
    is in-built to this component.

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signals
    | weights - current value matrix of synaptic efficacies
    | biases - current value vector of synaptic bias values

    Args:
        name: the string name of this cell

        shape: tuple specifying shape of this synaptic cable (usually a 2-tuple
            with number of inputs by number of outputs)

        weight_init: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use

        bias_init: a kernel to drive initialization of biases for this synaptic cable
            (Default: None, which turns off/disables biases)

        resist_scale: a fixed (resistance) scaling factor to apply to synaptic
            transform (Default: 1.), i.e., yields: out = ((W * in) * resist_scale) + bias

        p_conn: probability of a connection existing (default: 1.); setting
            this to < 1 and > 0. will result in a sparser synaptic structure
            (lower values yield sparse structure)
    """

    # Define Functions
    def __init__(self, name, shape, model_shape=(1,1), weight_forward=True, weight_init=None, bias_init=None,
                 resist_scale=1., p_conn=1., batch_size=1, **kwargs):
        super().__init__(name, **kwargs)

        self.batch_size = batch_size
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.model_shape = model_shape
        self.n_models = model_shape[0]                 # = ni = #of parents = n3
        self.model_patches = model_shape[1]              # = ci = #of children per parent = c3
        self.weight_forward = weight_forward

        ## Synapse meta-parameters
        self.sub_shape = shape[0], shape[1]*self.model_patches  ## shape of synaptic efficacy matrix           # = (in_dim, hid_dim) = (d3, d2)
        self.Rscale = resist_scale ## post-transformation scale factor
        self.shape = (self.sub_shape[0] * self.n_models, self.sub_shape[1] * self.n_models)

        if self.weight_forward == False:
            self.sub_shape = (self.sub_shape[1], self.sub_shape[0])
            self.shape = (self.shape[1], self.shape[0])

        ## Set up synaptic weight values
        tmp_key, *subkeys = random.split(self.key.value, 4)
        if self.weight_init is None:
            info(self.name, "is using default weight initializer!")
            self.weight_init = {"dist": "uniform", "amin": 0.025, "amax": 0.8}


        if self.n_models>1 or self.model_patches>1:
            weights = initialize_params(subkeys[0], {"dist": "constant", "value": 0.}, self.shape, use_numpy=True)
            for i in range(self.n_models):
                weights[self.sub_shape[0] * i:
                        self.sub_shape[0] * (i + 1),
                        self.sub_shape[1] * i:
                        self.sub_shape[1] * (i + 1)] = initialize_params(subkeys[0],
                                                                 init_kernel=self.weight_init,
                                                                 shape=self.sub_shape,
                                                                 use_numpy=True)
        else:
            weights = initialize_params(subkeys[0], self.weight_init, self.shape)
        if 0. < p_conn < 1.: ## only non-zero and <1 probs allowed
            mask = random.bernoulli(subkeys[1], p=p_conn, shape=self.shape)
            weights = weights * mask ## sparsify matrix

        self.batch_size = 1
        ## Compartment setup
        preVals = jnp.zeros((self.batch_size, self.shape[0]))
        postVals = jnp.zeros((self.batch_size, self.shape[1]))
        self.inputs = Compartment(preVals)
        self.outputs = Compartment(postVals)
        self.weights = Compartment(weights)
        ## Set up (optional) bias values
        if self.bias_init is None:
            info(self.name, "is using default bias value of zero (no bias "
                            "kernel provided)!")
        self.biases = Compartment(initialize_params(subkeys[2], bias_init,
                                                    (1, self.shape[1]))
                                  if bias_init else 0.0)

    @staticmethod
    def _advance_state(Rscale, inputs, weights, biases):
        outputs = (jnp.matmul(inputs, weights) * Rscale) + biases
        return outputs

    @resolver(_advance_state)
    def advance_state(self, outputs):
        self.outputs.set(outputs)

    @staticmethod
    def _reset(batch_size, shape):
        preVals = jnp.zeros((batch_size, shape[0]))
        postVals = jnp.zeros((batch_size, shape[1]))
        inputs = preVals
        outputs = postVals
        return inputs, outputs

    @resolver(_reset)
    def reset(self, inputs, outputs):
        self.inputs.set(inputs)
        self.outputs.set(outputs)

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        if self.bias_init != None:
            jnp.savez(file_name, weights=self.weights.value,
                      biases=self.biases.value)
        else:
            jnp.savez(file_name, weights=self.weights.value)

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.weights.set(data['weights'])
        if "biases" in data.keys():
            self.biases.set(data['biases'])

    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "DenseSynapse - performs a synaptic transformation "
                            "of inputs to produce  output signals (e.g., a "
                            "scaled linear multivariate transformation)"
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values"},
            "states":
                {"weights": "Synapse efficacy/strength parameter values",
                 "biases": "Base-rate/bias parameter values",
                 "key": "JAX PRNG key"},
            "outputs":
                {"outputs": "Output of synaptic transformation"},
        }
        hyperparams = {
            "shape": "Shape of synaptic weight value matrix; number inputs x number outputs",
            "batch_size": "Batch size dimension of this component",
            "weight_init": "Initialization conditions for synaptic weight (W) values",
            "bias_init": "Initialization conditions for bias/base-rate (b) values",
            "resist_scale": "Resistance level scaling factor (Rscale); applied to output of transformation",
            "p_conn": "Probability of a connection existing (otherwise, it is masked to zero)"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [W * inputs] * Rscale + b",
                "hyperparameters": hyperparams}
        return info

    def __repr__(self):
        comps = [varname for varname in dir(self) if Compartment.is_compartment(getattr(self, varname))]
        maxlen = max(len(c) for c in comps) + 5
        lines = f"[{self.__class__.__name__}] PATH: {self.name}\n"
        for c in comps:
            stats = tensorstats(getattr(self, c).value)
            if stats is not None:
                line = [f"{k}: {v}" for k, v in stats.items()]
                line = ", ".join(line)
            else:
                line = "None"
            lines += f"  {f'({c})'.ljust(maxlen)}{line}\n"
        return lines

if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        Wab = DenseSynapse("Wab", (2, 3))
    print(Wab)
