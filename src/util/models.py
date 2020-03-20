import numpy as np

import netket as nk
from netket.layer import SumOutput
from netket.layer import FullyConnected
from netket.layer import Lncosh
from netket.hilbert import Spin
from netket.graph import Hypercube
from netket.machine import FFNN
from netket.layer import ConvolutionalHypercube




def build_model_flowket(cf, input_shape):
    conditional_log_probs_model = None
    if cf.model_name == "rbm":
        from ..architectures.rbm import rbm
        model = rbm(cf, input_shape)
    elif cf.model_name == "drbm":
        from ..architectures.drbm import drbm
        model = drbm(cf, input_shape)
    elif cf.model_name == "ar1":
        from ..architectures.ar1 import ar
        model, conditional_log_probs_model = ar(cf, input_shape)
    elif cf.model_name == "ar2":
        from ..architectures.ar2 import ar
        model, conditional_log_probs_model = ar(cf, input_shape)
    elif cf.model_name == "my_rbm":
        from ..architectures.my_rbm import my_rbm
        model = my_rbm(cf, input_shape)
    model.summary()
    return model, conditional_log_probs_model


def build_model_netket(cf, hilbert):
    if cf.model_name == "rbm":
        model = nk.machine.RbmSpin(alpha=1, hilbert=hilbert)
    elif cf.model_name == "mlp1":
        input_size = np.prod(cf.input_size)
        layers = (FullyConnected(input_size=input_size,output_size=input_size,use_bias=True),
                  Lncosh(input_size=input_size),
                  SumOutput(input_size=input_size))
        model = FFNN(hilbert, layers)
    elif cf.model_name == "mlp2":
        input_size = np.prod(cf.input_size)
        layers = (FullyConnected(input_size=input_size,output_size=input_size*2,use_bias=True),
                  FullyConnected(input_size=input_size*2,output_size=input_size,use_bias=True),
                  Lncosh(input_size=input_size),
                  SumOutput(input_size=input_size))
        model = FFNN(hilbert, layers)
    elif cf.model_name == "conv_net":
        input_size = cf.input_size
        dim = len(input_size)
        length = input_size[0]
        assert dim == 2
        assert input_size[0] == input_size[1]
        layers = (ConvolutionalHypercube(length=length, n_dim=dim, input_channels=1, output_channels=1, stride=1, kernel_length=3, use_bias=True),
                  FullyConnected(input_size=np.prod(input_size),output_size=np.prod(input_size),use_bias=True),
                  Lncosh(input_size=np.prod(input_size)),
                  SumOutput(input_size=np.prod(input_size)))
        model = FFNN(hilbert, layers)
    return model


def load_model(cf, model, loadpath):
    if cf.num_gpu<2:
        bad_state_dict = torch.load(loadpath,map_location='cpu')
        correct_state_dict = {re.sub(r'^module\.', '', k): v for k, v in
                                bad_state_dict.items()}
        model.load_state_dict(correct_state_dict)
    else:
        model.load_state_dict(torch.load(loadpath))
    model.eval()
    model.zero_grad()
    return model

