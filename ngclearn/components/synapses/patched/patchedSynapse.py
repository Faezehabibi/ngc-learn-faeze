import matplotlib.pyplot as plt
from jax import random, numpy as jnp, jit
from ngclearn import resolver, Component, Compartment
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn.utils import tensorstats
from ngclearn.utils.weight_distribution import initialize_params
from ngcsimlib.logger import info
import math


"""
       𝑳𝒾                     𝑳𝑗       
       ⬇︎                     ⬇︎          
    𝒏𝒾 = 3, 𝑫𝒾                  𝑫𝑗    
    𝒅𝒾 = 𝑫𝒾 / 𝒏𝒾            𝒅𝑗 = 𝑫𝑗 / 𝒏𝒾

        ⎯[𝑤𝒾𝑗¹      𝟘       𝟘 ]⎯      
  Z𝒾    ⎯[ 𝟘      𝑤𝒾𝑗²      𝟘 ]⎯  Z𝑗
        ⎯[ 𝟘       𝟘      𝑤𝒾𝑗³]⎯ 

"""


def create_multi_patch_synapses(key, shape, n_sub_models, sub_stride, weight_init):
    sub_shape = (shape[0] // n_sub_models, shape[1] // n_sub_models)
    di, dj = sub_shape
    si, sj = sub_stride

    weight_shape = ((n_sub_models * di) + 2 * si, (n_sub_models * dj) + 2 * sj)
    weights = initialize_params(key[2], {"dist": "constant", "value": 0.}, weight_shape, use_numpy=True)

    for i in range(n_sub_models):
        start_i = i * di
        end_i = (i + 1) * di + 2 * si
        start_j = i * dj
        end_j = (i + 1) * dj + 2 * sj

        shape_ = (end_i - start_i, end_j - start_j) # (di + 2 * si, dj + 2 * sj)

        weights[start_i : end_i,
                start_j : end_j] = initialize_params(key[2],
                                                     init_kernel=weight_init,
                                                     shape=shape_,
                                                     use_numpy=True)
    if si!=0:
        weights[:si,:] = 0.
        weights[-si:,:] = 0.
    if sj!=0:
        weights[:,:sj] = 0.
        weights[:, -sj:] = 0.

    return weights



class PatchedSynapse(JaxComponent): ## base patched synaptic cable
    # Define Functions
    def __init__(self, name, shape, n_sub_models, stride_shape=(0,0), w_mask=None, weight_init=None, bias_init=None,
                 resist_scale=1., p_conn=1., batch_size=1, **kwargs):
        super().__init__(name, **kwargs)

        self.Rscale = resist_scale
        self.batch_size = batch_size
        self.weight_init = weight_init
        self.bias_init = bias_init

        self.n_sub_models = n_sub_models
        self.sub_stride = stride_shape

        tmp_key, *subkeys = random.split(self.key.value, 4)
        if self.weight_init is None:
            info(self.name, "is using default weight initializer!")
            self.weight_init = {"dist": "fan_in_gaussian"}

        weights = create_multi_patch_synapses(key=subkeys, shape=shape, n_sub_models=self.n_sub_models, sub_stride=self.sub_stride,
                                              weight_init=self.weight_init)

        self.w_mask = jnp.where(weights!=0, 1, 0)
        self.sub_shape = (shape[0]//n_sub_models, shape[1]//n_sub_models)

        self.shape = weights.shape
        self.sub_shape = self.sub_shape[0]+(2*self.sub_stride[0]), self.sub_shape[1]+(2*self.sub_stride[1])

        if 0. < p_conn < 1.: ## only non-zero and <1 probs allowed
            mask = random.bernoulli(subkeys[1], p=p_conn, shape=self.shape)
            weights = weights * mask ## sparsify matrix

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
            "synapse_type": "PatchedSynapse - performs a synaptic transformation "
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
            "shape": "Overall shape of synaptic weight value matrix; number inputs x number outputs",
            "n_sub_models": "The number of submodels in each layer",
            "stride_shape": "Stride shape of overlapping synaptic weight value matrix",
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
        Wab = PatchedSynapse("Wab", (9, 30), 3)
    print(Wab)
    plt.imshow(Wab.weights.value, cmap='gray')
    plt.show()
