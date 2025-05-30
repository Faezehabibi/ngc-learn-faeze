from jax import random, numpy as jnp, jit
from ngclearn.components.jaxComponent import JaxComponent
from ngcsimlib.compilers.process import transition
from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment

from ngclearn.utils.weight_distribution import initialize_params
from ngcsimlib.logger import info
from ngclearn.utils import tensorstats
import ngclearn.utils.weight_distribution as dist
from ngclearn.components.synapses.convolution.ngcconv import deconv2d

class DeconvSynapse(JaxComponent): ## base-level deconvolutional cable
    """
    A base deconvolutional (transposed convolutional) synaptic cable.

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signals
    | filters - current value tensor of filter/kernel efficacies
    | biases - current base-rate/bias efficacies

    Args:
        name: the string name of this cell

        x_shape: 2d shape of input map signal (component currently assumess a square input maps)

        shape: tuple specifying shape of this synaptic cable (usually a 4-tuple
            with number `filter height x filter width x input channels x number output channels`);
            note that currently filters/kernels are assumed to be square
            (kernel.width = kernel.height)

        filter_init: a kernel to drive initialization of this synaptic cable's
            filter values

        bias_init: kernel to drive initialization of bias/base-rate values
            (Default: None, which turns off/disables biases)

        stride: length/size of stride

        padding: pre-operator padding to use -- "VALID" (none), "SAME"

        resist_scale: a fixed (resistance) scaling factor to apply to synaptic
            transform (Default: 1.), i.e., yields: out = ((W @.T Rscale) * in) + b
            where `@.T` denotes deconvolution

        batch_size: batch size dimension of this component
    """

    # Define Functions
    def __init__(
            self, name, shape, x_shape, filter_init=None, bias_init=None, stride=1, padding=None, resist_scale=1.,
            batch_size=1, **kwargs
    ):
        super().__init__(name, **kwargs)

        self.filter_init = filter_init
        self.bias_init = bias_init

        ## Synapse meta-parameters
        self.shape = shape ## shape of synaptic filter tensor
        x_size, x_size = x_shape
        self.x_size = x_size
        self.Rscale = resist_scale ## post-transformation scale factor
        self.padding = padding
        self.stride = stride

        ####################### Set up padding arguments #######################
        k_size, k_size, n_in_chan, n_out_chan = shape
        self.pad_args = None

        ######################### set up compartments ##########################
        tmp_key, *subkeys = random.split(self.key.value, 4)
        weights = dist.initialize_params(subkeys[0], filter_init,
                                         shape)  ## filter tensor
        self.batch_size = batch_size # 1
        ## Compartment setup and shape computation
        _x = jnp.zeros((self.batch_size, x_size, x_size, n_in_chan))
        _d = deconv2d(_x, weights, stride_size=stride, padding=padding) * 0
        self.in_shape = _x.shape
        self.out_shape = _d.shape
        self.inputs = Compartment(jnp.zeros(self.in_shape))
        self.outputs = Compartment(jnp.zeros(self.out_shape))
        self.weights = Compartment(weights)
        if self.bias_init is None:
            info(self.name, "is using default bias value of zero (no bias "
                            "kernel provided)!")
        self.biases = Compartment(dist.initialize_params(subkeys[2], bias_init,
                                                         (1, shape[1]))
                                  if bias_init else 0.0)

    @transition(output_compartments=["outputs"])
    @staticmethod
    def advance_state(Rscale, padding, stride, weights, biases, inputs):
        _x = inputs
        out = deconv2d(_x, weights, stride_size=stride, padding=padding) * Rscale + biases
        return out

    @transition(output_compartments=["inputs", "outputs"])
    @staticmethod
    def reset(in_shape, out_shape):
        preVals = jnp.zeros(in_shape)
        postVals = jnp.zeros(out_shape)
        inputs = preVals
        outputs = postVals
        return inputs, outputs

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
            "synapse type": "DeconvSynapse - performs a synaptic deconvolution (@.T) of "
                            "inputs to produce output signals"
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values"},
            "states":
                {"filters": "Synaptic filter parameter values",
                 "biases": "Base-rate/bias parameter values",
                 "key": "JAX PRNG key"},
            "outputs":
                {"outputs": "Output of synaptic transformation"},
        }
        hyperparams = {
            "shape": "Shape of synaptic filter value matrix; `kernel width` x `kernel height` "
                     "x `number input channels` x `number output channels`",
            "x_shape": "Shape of any single incoming/input feature map",
            "filter_init": "Initialization conditions for synaptic filter (K) values",
            "bias_init": "Initialization conditions for bias/base-rate (b) values",
            "resist_scale": "Resistance level output scaling factor (R)",
            "stride": "length / size of stride",
            "padding": "pre-operator padding to use, i.e., `VALID` `SAME`"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [K @.T inputs] * R + b",
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
