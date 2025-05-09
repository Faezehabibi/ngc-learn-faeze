from jax import random, numpy as jnp, jit
from ngcsimlib.compilers.process import transition
from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment

from .convSynapse import ConvSynapse
from ngclearn.utils.weight_distribution import initialize_params
from ngcsimlib.logger import info
from ngclearn.utils import tensorstats
import ngclearn.utils.weight_distribution as dist
from ngclearn.components.synapses.convolution.ngcconv import (_conv_same_transpose_padding,
                                                              _conv_valid_transpose_padding)
from ngclearn.components.synapses.convolution.ngcconv import (conv2d, _calc_dX_conv,
                                                              _calc_dK_conv, calc_dX_conv,
                                                              calc_dK_conv)
from ngclearn.utils.optim import get_opt_init_fn, get_opt_step_fn

class HebbianConvSynapse(ConvSynapse): ## Hebbian-evolved convolutional cable
    """
    A synaptic convolutional cable that adjusts its efficacies via a two-factor
    Hebbian adjustment rule.

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signals (transformation induced by filters)
    | filters - current value matrix of synaptic filter efficacies
    | biases - current value vector of synaptic bias values
    | key - JAX PRNG key
    | --- Synaptic Plasticity Compartments: ---
    | pre - pre-synaptic signal to drive first term of Hebbian update (takes in external signals)
    | post - post-synaptic signal to drive 2nd term of Hebbian update (takes in external signals)
    | dWeights - delta tensor containing changes to be applied to synaptic filter efficacies
    | dBiases - delta tensor containing changes to be applied to bias values
    | dInputs - delta tensor containing back-transmitted signal values ("backpropagating pulse")
    | opt_params - locally-embedded optimizer statisticis (e.g., Adam 1st/2nd moments if adam is used)

    Args:
        name: the string name of this cell

        x_shape: 2d shape of input map signal (component currently assumess a square input maps)

        shape: tuple specifying shape of this synaptic cable (usually a 4-tuple
            with number `filter height x filter width x input channels x number output channels`);
            note that currently filters/kernels are assumed to be square
            (kernel.width = kernel.height)

        eta: global learning rate (default: 0)

        filter_init: a kernel to drive initialization of this synaptic cable's
            filter values

        bias_init: kernel to drive initialization of bias/base-rate values
            (Default: None, which turns off/disables biases)

        stride: length/size of stride

        padding: pre-operator padding to use -- "VALID" (none), "SAME"

        resist_scale: a fixed (resistance) scaling factor to apply to synaptic
            transform (Default: 1.), i.e., yields: out = ((K @ in) * resist_scale) + b
            where `@` denotes convolution

        w_bound: maximum weight to softly bound this cable's value matrix to; if
            set to 0, then no synaptic value bounding will be applied

        is_nonnegative: enforce that synaptic efficacies are always non-negative
            after each synaptic update (if False, no constraint will be applied)

        w_decay: degree to which (L2) synaptic weight decay is applied to the
            computed Hebbian adjustment (Default: 0); note that decay is not
            applied to any configured biases

        sign_value: multiplicative factor to apply to final synaptic update before
            it is applied to synapses; this is useful if gradient descent style
            optimization is required (as Hebbian rules typically yield
            adjustments for ascent)

        optim_type: optimization scheme to physically alter synaptic values
            once an update is computed (Default: "sgd"); supported schemes
            include "sgd" and "adam"

            :Note: technically, if "sgd" or "adam" is used but `signVal = 1`,
                then the ascent form of each rule is employed (signVal = -1) or
                a negative learning rate will mean a descent form of the
                `optim_scheme` is being employed

        batch_size: batch size dimension of this component
    """

    # Define Functions
    def __init__(self, name, shape, x_shape, eta=0., filter_init=None, bias_init=None,
                 stride=1, padding=None, resist_scale=1., w_bound=0.,
                 is_nonnegative=False, w_decay=0., sign_value=1., optim_type="sgd",
                 batch_size=1, **kwargs):
        super().__init__(
            name, shape, x_shape=x_shape, filter_init=filter_init, bias_init=bias_init, resist_scale=resist_scale,
            stride=stride, padding=padding, batch_size=batch_size, **kwargs
        )

        self.eta = eta
        self.w_bounds = w_bound
        self.w_decay = w_decay  ## synaptic decay
        self.is_nonnegative = is_nonnegative
        self.sign_value = sign_value
        ## optimization / adjustment properties (given learning dynamics above)
        self.opt = get_opt_step_fn(optim_type, eta=self.eta)

        ######################### set up compartments ##########################
        ## Compartment setup and shape computation
        self.dWeights = Compartment(self.weights.value * 0)
        self.dInputs = Compartment(jnp.zeros(self.in_shape))
        self.dBiases = Compartment(self.biases.value * 0)
        self.pre = Compartment(jnp.zeros(self.in_shape))
        self.post = Compartment(jnp.zeros(self.out_shape))

        ########################################################################
        ## Shape error correction -- do shape correction inference for local updates
        self._init(self.batch_size, self.x_size, self.shape, self.stride,
                   self.padding, self.pad_args, self.weights)
        self.antiPad = None
        k_size, k_size, n_in_chan, n_out_chan = self.shape
        if padding == "SAME":
            self.antiPad = _conv_same_transpose_padding(self.post.value.shape[1],
                                                        self.x_size, k_size, stride)
        elif padding == "VALID":
            self.antiPad = _conv_valid_transpose_padding(self.post.value.shape[1],
                                                         self.x_size, k_size, stride)

        ########################################################################

        ## set up outer optimization compartments
        self.opt_params = Compartment(get_opt_init_fn(optim_type)(
            [self.weights.value, self.biases.value]
            if bias_init else [self.weights.value]))

    def _init(self, batch_size, x_size, shape, stride, padding, pad_args, weights):
        k_size, k_size, n_in_chan, n_out_chan = shape
        _x = jnp.zeros((batch_size, x_size, x_size, n_in_chan))
        _d = conv2d(_x, weights.value, stride_size=stride, padding=padding) * 0
        _dK = _calc_dK_conv(_x, _d, stride_size=stride, padding=pad_args)
        ## get filter update correction
        dx = _dK.shape[0] - weights.value.shape[0]
        dy = _dK.shape[1] - weights.value.shape[1]
        self.delta_shape = (max(dx, 0), max(dy, 0))
        ## get input update correction
        _dx = _calc_dX_conv(weights.value, _d, stride_size=stride,
                            anti_padding=pad_args)
        dx = (_dx.shape[1] - _x.shape[1])
        dy = (_dx.shape[2] - _x.shape[2])
        self.x_delta_shape = (dx, dy)

    @staticmethod
    def _compute_update(
            sign_value, w_decay, bias_init, stride, pad_args, delta_shape, pre, post, weights
    ): ## synaptic kernel adjustment calculation co-routine
        ## compute adjustment to filters
        dWeights = calc_dK_conv(pre, post, delta_shape=delta_shape, stride_size=stride, padding=pad_args)
        dWeights = dWeights * sign_value
        if w_decay > 0.:  ## apply synaptic decay
            dWeights = dWeights - weights * w_decay
        ## compute adjustment to base-rates (if applicable)
        dBiases = 0.  # jnp.zeros((1,1))
        if bias_init != None:
            dBiases = jnp.sum(post, axis=0, keepdims=True) * sign_value
        return dWeights, dBiases

    @transition(output_compartments=["opt_params", "weights", "biases", "dWeights", "dBiases"])
    @staticmethod
    def evolve(
            opt, sign_value, w_decay, w_bounds, is_nonnegative, bias_init, stride, pad_args, delta_shape, pre, post,
            weights, biases, opt_params
    ):
        ## calc dFilters / dBiases - update to filters and biases
        dWeights, dBiases = HebbianConvSynapse._compute_update(
            sign_value, w_decay, bias_init, stride, pad_args, delta_shape, pre, post, weights
        )
        if bias_init != None:
            opt_params, [weights, biases] = opt(opt_params, [weights, biases], [dWeights, dBiases])
        else: ## ignore dBiases since no biases configured
            opt_params, [weights] = opt(opt_params, [weights], [dWeights])

        ## apply any enforced filter constraints
        if w_bounds > 0.:
            if is_nonnegative:
                weights = jnp.clip(weights, 0., w_bounds)
            else:
                weights = jnp.clip(weights, -w_bounds, w_bounds)
        return opt_params, weights, biases, dWeights, dBiases

    @transition(output_compartments=["dInputs"])
    @staticmethod
    def backtransmit(
            sign_value, x_size, shape, stride, padding, x_delta_shape, antiPad, post, weights
    ): ## action-backpropagating routine
        ## calc dInputs - adjustment w.r.t. input signal
        k_size, k_size, n_in_chan, n_out_chan = shape
        # antiPad = None
        # if padding == "SAME":
        #     antiPad = _conv_same_transpose_padding(post.shape[1], x_size,
        #                                            k_size, stride)
        # elif padding == "VALID":
        #     antiPad = _conv_valid_transpose_padding(post.shape[1], x_size,
        #                                             k_size, stride)
        dInputs = calc_dX_conv(weights, post, delta_shape=x_delta_shape, stride_size=stride, anti_padding=antiPad)
        ## flip sign of back-transmitted signal (if applicable)
        dInputs = dInputs * sign_value
        return dInputs

    @transition(output_compartments=["inputs", "outputs", "pre", "post", "dInputs"])
    @staticmethod
    def reset(in_shape, out_shape):
        preVals = jnp.zeros(in_shape)
        postVals = jnp.zeros(out_shape)
        inputs = preVals
        outputs = postVals
        pre = preVals
        post = postVals
        dInputs = preVals
        return inputs, outputs, pre, post, dInputs

    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "HebbianConvSynapse - performs a synaptic convolution "
                            "(@) of inputs  to produce output signals; synaptic "
                            "filters are adjusted via two-term/factor Hebbian "
                            "adjustment"
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values",
                 "pre": "Pre-synaptic statistic for Hebb rule (z_j)",
                 "post": "Post-synaptic statistic for Hebb rule (z_i)"},
            "states":
                {"filters": "Synaptic filter parameter values",
                 "biases": "Base-rate/bias parameter values",
                 "key": "JAX PRNG key"},
            "analytics":
                {"dWeights": "Synaptic filter value adjustment 4D-tensor produced at time t",
                 "dBiases": "Synaptic bias/base-rate value adjustment 3D-tensor produced at time t"},
            "outputs":
                {"outputs": "Output of synaptic/filter transformation",
                 "dInputs": "Tensor containing back-transmitted signal values; backpropagating pulse"},
        }
        hyperparams = {
            "shape": "Shape of synaptic filter value matrix; `kernel width` x `kernel height` "
                     "x `number input channels` x `number output channels`",
            "x_shape": "Shape of any single incoming/input feature map",
            "filter_init": "Initialization conditions for synaptic filter (K) values",
            "bias_init": "Initialization conditions for bias/base-rate (b) values",
            "resist_scale": "Resistance level output scaling factor (R)",
            "stride": "length / size of stride",
            "padding": "pre-operator padding to use, i.e., `VALID` `SAME`",
            "is_nonnegative": "Should filters be constrained to be non-negative post-updates?",
            "sign_value": "Scalar `flipping` constant -- changes direction to Hebbian descent if < 0",
            "eta": "Global (fixed) learning rate",
            "w_bound": "Soft synaptic bound applied to filters post-update",
            "w_decay": "Synaptic filter decay term",
            "optim_type": "Optimization scheme to used for adjusting synapses"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [K @ inputs] * R + b; "
                            "dW_{ij}/dt = eta * [(z_j * q_pre) * (z_i * q_post)] - W_{ij} * w_decay",
                "hyperparameters": hyperparams}
        return info

