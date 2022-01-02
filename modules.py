import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import math
import numpy as np
from functools import partial
from torch import nn
import copy


def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_out')


def init_weights_selu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1/math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277)/math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def init_weights_uniform(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')


def sine_init(m, w0=30):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6/num_input)/w0, np.sqrt(6/num_input)/w0)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1/num_input, 1/num_input)


class FirstSine(nn.Module):
    def __init__(self, w0=20):
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, input):
        return torch.sin(self.w0*input)


class Sine(nn.Module):
    def __init__(self, w0=20):
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, input):
        return torch.sin(self.w0*input)


class MSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()
        self.cst = torch.log(torch.tensor(2.))

    def forward(self, input):
        return self.softplus(input)-self.cst


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input*torch.sigmoid(input)


def mfn_weights_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6/num_input), np.sqrt(6/num_input))


class MFNBase(nn.Module):

    def __init__(self, hidden_size, out_size, n_layers, weight_scale,
                 bias=True, output_act=False):
        super().__init__()

        self.linear = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias) for _ in range(n_layers)]
        )

        self.output_linear = nn.Linear(hidden_size, out_size)

        self.output_act = output_act

        self.linear.apply(mfn_weights_init)
        self.output_linear.apply(mfn_weights_init)

    def forward(self, model_input):

        input_dict = {key: input.clone().detach().requires_grad_(True)
                      for key, input in model_input.items()}
        coords = input_dict['coords']

        out = self.filters[0](coords)
        for i in range(1, len(self.filters)):
            out = self.filters[i](coords) * self.linear[i - 1](out)
        out = self.output_linear(out)

        if self.output_act:
            out = torch.sin(out)

        return {'model_in': input_dict, 'model_out': {'output': out}}


class FourierLayer(nn.Module):

    def __init__(self, in_features, out_features, weight_scale, quantization_interval=2*np.pi):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

        r = 2*weight_scale[0] / quantization_interval
        assert math.isclose(r, round(r)), \
               'weight_scale should be divisible by quantization interval'

        # sample discrete uniform distribution of frequencies
        for i in range(self.linear.weight.data.shape[1]):
            init = torch.randint_like(self.linear.weight.data[:, i],
                                      0, int(2*weight_scale[i] / quantization_interval)+1)
            init = init * quantization_interval - weight_scale[i]
            self.linear.weight.data[:, i] = init

        self.linear.weight.requires_grad = False
        self.linear.bias.data.uniform_(-np.pi, np.pi)
        return

    def forward(self, x):
        return torch.sin(self.linear(x))


class BACON(MFNBase):
    def __init__(self,
                 in_size,
                 hidden_size,
                 out_size,
                 hidden_layers=3,
                 weight_scale=1.0,
                 bias=True,
                 output_act=False,
                 frequency=(128, 128),
                 quantization_interval=2*np.pi,  # assumes data range [-.5,.5]
                 centered=True,
                 input_scales=None,
                 output_layers=None,
                 is_sdf=False,
                 reuse_filters=False,
                 **kwargs):

        super().__init__(hidden_size, out_size, hidden_layers,
                         weight_scale, bias, output_act)

        self.quantization_interval = quantization_interval
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.centered = centered
        self.frequency = frequency
        self.is_sdf = is_sdf
        self.reuse_filters = reuse_filters
        self.in_size = in_size

        # we need to multiply by this to be able to fit the signal
        input_scale = [round((np.pi * freq / (hidden_layers + 1))
                       / quantization_interval) * quantization_interval for freq in frequency]

        self.filters = nn.ModuleList([
                FourierLayer(in_size, hidden_size, input_scale,
                             quantization_interval=quantization_interval)
                for i in range(hidden_layers + 1)])

        print(self)

    def forward_mfn(self, input_dict):
        if 'coords' in input_dict:
            coords = input_dict['coords']
        elif 'ray_samples' in input_dict:
            if self.in_size > 3:
                coords = torch.cat((input_dict['ray_samples'], input_dict['ray_orientations']), dim=-1)
            else:
                coords = input_dict['ray_samples']

        if self.reuse_filters:
            filter_outputs = 3 * [self.filters[2](coords), ] + \
                             2 * [self.filters[4](coords), ] + \
                             2 * [self.filters[6](coords), ] + \
                             2 * [self.filters[8](coords), ]

            out = filter_outputs[0]
            for i in range(1, len(self.filters)):
                out = filter_outputs[i] * self.linear[i - 1](out)
        else:
            out = self.filters[0](coords)
            for i in range(1, len(self.filters)):
                out = self.filters[i](coords) * self.linear[i - 1](out)

        out = self.output_linear(out)

        if self.output_act:
            out = torch.sin(out)

        return out

    def forward(self, model_input, mode=None, integral_dim=None):

        out = {'output': self.forward_mfn(model_input)}

        if self.is_sdf:
            return {'model_in': model_input['coords'],
                    'model_out': out['output']}

        return {'model_in': model_input, 'model_out': out}


class MultiscaleBACON(MFNBase):
    def __init__(self,
                 in_size,
                 hidden_size,
                 out_size,
                 hidden_layers=3,
                 weight_scale=1.0,
                 bias=True,
                 output_act=False,
                 frequency=(128, 128),
                 quantization_interval=2*np.pi,
                 centered=True,
                 is_sdf=False,
                 input_scales=None,
                 output_layers=None,
                 reuse_filters=False):

        super().__init__(hidden_size, out_size, hidden_layers,
                         weight_scale, bias, output_act)

        self.quantization_interval = quantization_interval
        self.hidden_layers = hidden_layers
        self.centered = centered
        self.is_sdf = is_sdf
        self.frequency = frequency
        self.output_layers = output_layers
        self.reuse_filters = reuse_filters
        self.stop_after = None

        # we need to multiply by this to be able to fit the signal
        if input_scales is None:
            input_scale = [round((np.pi * freq / (hidden_layers + 1))
                           / quantization_interval) * quantization_interval for freq in frequency]

            self.filters = nn.ModuleList([
                    FourierLayer(in_size, hidden_size, input_scale,
                                 quantization_interval=quantization_interval)
                    for i in range(hidden_layers + 1)])
        else:
            if len(input_scales) != hidden_layers+1:
                raise ValueError('require n+1 scales for n hidden_layers')
            input_scale = [[round((np.pi * freq * scale) / quantization_interval) * quantization_interval
                           for freq in frequency] for scale in input_scales]

            self.filters = nn.ModuleList([
                           FourierLayer(in_size, hidden_size, input_scale[i],
                                        quantization_interval=quantization_interval)
                           for i in range(hidden_layers + 1)])

        # linear layers to extract intermediate outputs
        self.output_linear = nn.ModuleList([nn.Linear(hidden_size, out_size) for i in range(len(self.filters))])
        self.output_linear.apply(mfn_weights_init)

        # if outputs layers is None, output at every possible layer
        if self.output_layers is None:
            self.output_layers = np.arange(1, len(self.filters))

        print(self)

    def layer_forward(self, coords, filter_outputs, specified_layers,
                      get_feature, continue_layer, continue_feature):
        """ for multiscale SDF extraction """

        # hardcode the 8 layer network that we use for all sdf experiments
        filter_ind_dict = [2, 2, 2, 4, 4, 6, 6, 8, 8]
        outputs = []

        if continue_feature is None:
            assert(continue_layer == 0)
            out = self.filters[filter_ind_dict[0]](coords)
            filter_output_dict = {filter_ind_dict[0]: out}
        else:
            out = continue_feature
            filter_output_dict = {}

        for i in range(continue_layer+1, len(self.filters)):
            if filter_ind_dict[i] not in filter_output_dict.keys():
                filter_output_dict[filter_ind_dict[i]] = self.filters[filter_ind_dict[i]](coords)
            out = filter_output_dict[filter_ind_dict[i]] * self.linear[i - 1](out)

            if i in self.output_layers and i == specified_layers:
                if get_feature:
                    outputs.append([self.output_linear[i](out), out])
                else:
                    outputs.append(self.output_linear[i](out))
                return outputs

        return outputs

    def forward(self, model_input, specified_layers=None, get_feature=False,
                continue_layer=0, continue_feature=None):

        if self.is_sdf:
            model_input = {key: input.clone().detach().requires_grad_(True)
                           for key, input in model_input.items()}

        if 'coords' in model_input:
            coords = model_input['coords']
        elif 'ray_samples' in model_input:
            coords = model_input['ray_samples']

        outputs = []
        if self.reuse_filters:

            # which layers to reuse
            if len(self.filters) < 9:
                filter_outputs = 2 * [self.filters[0](coords), ] + \
                        (len(self.filters)-2) * [self.filters[-1](coords), ]
            else:
                filter_outputs = 3 * [self.filters[2](coords), ] + \
                                 2 * [self.filters[4](coords), ] + \
                                 2 * [self.filters[6](coords), ] + \
                                 2 * [self.filters[8](coords), ]

            # multiscale sdf extractions (evaluate only some layers)
            if specified_layers is not None:
                outputs = self.layer_forward(coords, filter_outputs, specified_layers,
                                             get_feature, continue_layer, continue_feature)

            # evaluate all layers
            else:
                out = filter_outputs[0]
                for i in range(1, len(self.filters)):
                    out = filter_outputs[i] * self.linear[i - 1](out)

                    if i in self.output_layers:
                        outputs.append(self.output_linear[i](out))
                        if self.stop_after is not None and len(outputs) > self.stop_after:
                            break

        # no layer reuse
        else:
            out = self.filters[0](coords)
            for i in range(1, len(self.filters)):
                out = self.filters[i](coords) * self.linear[i - 1](out)

                if i in self.output_layers:
                    outputs.append(self.output_linear[i](out))
                    if self.stop_after is not None and len(outputs) > self.stop_after:
                        break

        if self.is_sdf:  # convert dtype
            return {'model_in': model_input['coords'],
                    'model_out': outputs}  # outputs is a list of tensors

        return {'model_in': model_input, 'model_out': {'output': outputs}}


class MultiscaleCoordinateNet(nn.Module):
    '''A canonical coordinate network'''
    def __init__(self, out_features=1, nl='sine', in_features=1,
                 hidden_features=256, num_hidden_layers=3,
                 w0=30, pe_scale=6, use_sigmoid=True, no_pe=False,
                 integrated_pe=False):

        super().__init__()

        self.nl = nl
        dims = in_features
        self.use_sigmoid = use_sigmoid
        self.no_pe = no_pe
        self.integrated_pe = integrated_pe

        if integrated_pe:
            self.pe = partial(IntegratedPositionalEncoding, L=pe_scale)
            in_features = int(2 * in_features * pe_scale)

        elif self.nl != 'sine' and not self.no_pe:
            in_features = in_features * hidden_features
            self.pe = FFPositionalEncoding(hidden_features, pe_scale, dims=dims)

        self.net = FCBlock(in_features=in_features,
                           out_features=out_features,
                           num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features,
                           outermost_linear=True,
                           nonlinearity=nl,
                           w0=w0).net

        if not integrated_pe:
            self.output_linear = nn.ModuleList([nn.Linear(hidden_features, out_features)
                                               for i in range(num_hidden_layers)])
            self.net = self.net[:-1]

        print(self)

    def net_forward(self, coords):
        outputs = []

        if self.use_sigmoid and self.nl != 'sine':
            def out_nl(x): return torch.sigmoid(x)
        else:
            def out_nl(x): return x

        # mipnerf baseline
        if self.integrated_pe:
            for c in coords:
                outputs.append(out_nl(self.net(c)))
        else:
            out = self.net[0](coords)
            outputs.append(self.output_linear[0](out))
            for i, n in enumerate(self.net[1:]):
                # run main branch
                out = n(out)

                # extract intermediate output
                outputs.append(out_nl(self.output_linear[i](out)))

        return outputs

    def forward(self, model_input):

        coords = model_input['coords']

        if self.integrated_pe:
            coords = [self.pe(coords, r) for r in model_input['radii']]
        elif self.nl != 'sine' and not self.no_pe:
            coords = self.pe(coords)

        output = self.net_forward(coords)

        return {'model_in': model_input, 'model_out': {'output': output}}


class CoordinateNet(nn.Module):
    '''A canonical coordinate network'''
    def __init__(self, out_features=1, nl='sine', in_features=1,
                 hidden_features=256, num_hidden_layers=3,
                 w0=30, pe_scale=5, use_sigmoid=True, no_pe=False,
                 is_sdf=False, **kwargs):

        super().__init__()

        self.nl = nl
        dims = in_features
        self.use_sigmoid = use_sigmoid
        self.no_pe = no_pe
        self.is_sdf = is_sdf

        if self.nl != 'sine' and not self.no_pe:
            in_features = hidden_features  # in_features * hidden_features

            self.pe = FFPositionalEncoding(hidden_features, pe_scale, dims=dims)

        self.net = FCBlock(in_features=in_features,
                           out_features=out_features,
                           num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features,
                           outermost_linear=True,
                           nonlinearity=nl,
                           w0=w0)
        print(self)

    def forward(self, model_input):

        coords = model_input['coords']

        if self.nl != 'sine' and not self.no_pe:
            coords_pe = self.pe(coords)
            output = self.net(coords_pe)
            if self.use_sigmoid:
                output = torch.sigmoid(output)
        else:
            output = self.net(coords)

        if self.is_sdf:
            return {'model_in': model_input, 'model_out': output}

        else:
            return {'model_in': model_input, 'model_out': {'output': output}}


def IntegratedPositionalEncoding(coords, radius, L=8):

    # adapted from mipnerf https://github.com/google/mipnerf
    def expected_sin(x, x_var):
        """Estimates mean and variance of sin(z), z ~ N(x, var)."""

        # When the variance is wide, shrink sin towards zero.
        y = torch.exp(-0.5 * x_var) * torch.sin(x)
        y_var = torch.clip(0.5 * (1 - torch.exp(-2 * x_var) * torch.cos(2 * x)) - y**2, 0)
        return y, y_var

    def integrated_pos_enc(x_coord, min_deg, max_deg):
        """Encode `x` with sinusoids scaled by 2^[min_deg:max_deg-1]."""

        x, x_cov_diag = x_coord
        scales = torch.tensor([2**i for i in range(int(min_deg), int(max_deg))], device=x.device)
        shape = list(x.shape[:-1]) + [-1]

        y = torch.reshape(x[..., None, :] * scales[:, None], shape)
        y_var = torch.reshape(x_cov_diag[..., None, :] * scales[:, None]**2, shape)

        return expected_sin(
            torch.cat([y, y + 0.5 * np.pi], dim=-1),
            torch.cat([y_var] * 2, dim=-1))[0]

    means = coords
    covs = (radius**2 / 4) * torch.ones((1, 2), device=coords.device).repeat(coords.shape[-2], 1)
    return integrated_pos_enc((means, covs), 0, L)


class FFPositionalEncoding(nn.Module):
    def __init__(self, embedding_size, scale, dims=2, gaussian=True):
        super().__init__()
        self.embedding_size = embedding_size
        self.scale = scale

        if gaussian:
            bvals = torch.randn(embedding_size // 2, dims) * scale
        else:
            bvals = 2.**torch.linspace(0, scale, embedding_size//2) - 1

            if dims == 1:
                bvals = bvals[:, None]

            elif dims == 2:
                bvals = torch.stack([bvals, torch.zeros_like(bvals)], dim=-1)
                bvals = torch.cat([bvals, torch.roll(bvals, 1, -1)], dim=0)

            else:
                tmp = (dims-1)*(torch.zeros_like(bvals),)
                bvals = torch.stack([bvals, *tmp], dim=-1)

                tmp = [torch.roll(bvals, i, -1) for i in range(1, dims)]
                bvals = torch.cat([bvals, *tmp], dim=0)

        avals = torch.ones((bvals.shape[0]))
        self.avals = nn.Parameter(avals, requires_grad=False)
        self.bvals = nn.Parameter(bvals, requires_grad=False)

    def forward(self, tensor) -> torch.Tensor:
        """
            Apply positional encoding to the input.
        """

        return torch.cat([self.avals * torch.sin((2.*np.pi*tensor) @ self.bvals.T),
                          self.avals * torch.cos((2.*np.pi*tensor) @ self.bvals.T)], dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, num_encoding_functions=6, include_input=True, log_sampling=True, normalize=False,
                 input_dim=3, gaussian_pe=False, gaussian_variance=38):
        super().__init__()
        self.num_encoding_functions = num_encoding_functions
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.normalize = normalize
        self.gaussian_pe = gaussian_pe
        self.normalization = None

        if self.gaussian_pe:
            # this needs to be registered as a parameter so that it is saved in the model state dict
            # and so that it is converted using .cuda(). Doesn't need to be trained though
            self.gaussian_weights = nn.Parameter(gaussian_variance * torch.randn(num_encoding_functions, input_dim),
                                                 requires_grad=False)

        else:
            self.frequency_bands = None
            if self.log_sampling:
                self.frequency_bands = 2.0 ** torch.linspace(
                    0.0,
                    self.num_encoding_functions - 1,
                    self.num_encoding_functions)
            else:
                self.frequency_bands = torch.linspace(
                    2.0 ** 0.0,
                    2.0 ** (self.num_encoding_functions - 1),
                    self.num_encoding_functions)

            if normalize:
                self.normalization = torch.tensor(1/self.frequency_bands)

    def forward(self, tensor) -> torch.Tensor:
        r"""Apply positional encoding to the input.

        Args:
            tensor (torch.Tensor): Input tensor to be positionally encoded.
            encoding_size (optional, int): Number of encoding functions used to compute
                a positional encoding (default: 6).
            include_input (optional, bool): Whether or not to include the input in the
                positional encoding (default: True).

        Returns:
        (torch.Tensor): Positional encoding of the input tensor.
        """

        encoding = [tensor] if self.include_input else []
        if self.gaussian_pe:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(torch.matmul(tensor, self.gaussian_weights.T)))
        else:
            for idx, freq in enumerate(self.frequency_bands):
                for func in [torch.sin, torch.cos]:
                    if self.normalization is not None:
                        encoding.append(self.normalization[idx]*func(tensor * freq))
                    else:
                        encoding.append(func(tensor * freq))

        # Special case, for no positional encoding
        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)


def layer_factory(layer_type, w0=30):
    layer_dict = \
        {
         'relu': (nn.ReLU(inplace=True), init_weights_uniform),
         'sigmoid': (nn.Sigmoid(), None),
         'fsine': (Sine(), first_layer_sine_init),
         'sine': (Sine(w0=w0), partial(sine_init, w0=w0)),
         'tanh': (nn.Tanh(), init_weights_xavier),
         'selu': (nn.SELU(inplace=True), init_weights_selu),
         'gelu': (nn.GELU(), init_weights_selu),
         'swish': (Swish(), init_weights_selu),
         'softplus': (nn.Softplus(), init_weights_normal),
         'msoftplus': (MSoftplus(), init_weights_normal),
         'elu': (nn.ELU(), init_weights_elu)
        }
    return layer_dict[layer_type]


class FCBlock(nn.Module):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''
    def __init__(self, in_features, out_features,
                 num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu',
                 weight_init=None, w0=30, set_bias=None,
                 dropout=0.0):
        super().__init__()

        self.first_layer_init = None
        self.dropout = dropout

        # Create hidden features list
        if not isinstance(hidden_features, list):
            num_hidden_features = hidden_features
            hidden_features = []
            for i in range(num_hidden_layers+1):
                hidden_features.append(num_hidden_features)
        else:
            num_hidden_layers = len(hidden_features)-1
        print(f"net_size={hidden_features}")

        # Create the net
        print(f"num_layers={len(hidden_features)}")
        if isinstance(nonlinearity, list):
            print(f"num_non_lin={len(nonlinearity)}")
            assert len(hidden_features) == len(nonlinearity), "Num hidden layers needs to " \
                                                              "match the length of the list of non-linearities"

            self.net = []
            self.net.append(nn.Sequential(
                nn.Linear(in_features, hidden_features[0]),
                layer_factory(nonlinearity[0])[0]
            ))
            for i in range(num_hidden_layers):
                self.net.append(nn.Sequential(
                    nn.Linear(hidden_features[i], hidden_features[i+1]),
                    layer_factory(nonlinearity[i+1])[0]
                ))

            if outermost_linear:
                self.net.append(nn.Sequential(
                    nn.Linear(hidden_features[-1], out_features),
                ))
            else:
                self.net.append(nn.Sequential(
                    nn.Linear(hidden_features[-1], out_features),
                    layer_factory(nonlinearity[-1])[0]
                ))
        elif isinstance(nonlinearity, str):
            nl, weight_init = layer_factory(nonlinearity, w0=w0)
            if(nonlinearity == 'sine'):
                first_nl = FirstSine(w0=w0)
                self.first_layer_init = first_layer_sine_init
            else:
                first_nl = nl

            if weight_init is not None:
                self.weight_init = weight_init

            self.net = []
            self.net.append(nn.Sequential(
                nn.Linear(in_features, hidden_features[0]),
                first_nl
            ))

            for i in range(num_hidden_layers):
                if(self.dropout > 0):
                    self.net.append(nn.Dropout(self.dropout))
                self.net.append(nn.Sequential(
                    nn.Linear(hidden_features[i], hidden_features[i+1]),
                    copy.deepcopy(nl)
                ))

            if (self.dropout > 0):
                self.net.append(nn.Dropout(self.dropout))
            if outermost_linear:
                self.net.append(nn.Sequential(
                    nn.Linear(hidden_features[-1], out_features),
                ))
            else:
                self.net.append(nn.Sequential(
                    nn.Linear(hidden_features[-1], out_features),
                    copy.deepcopy(nl)
                ))

        self.net = nn.Sequential(*self.net)

        if isinstance(nonlinearity, list):
            for layer_num, layer_name in enumerate(nonlinearity):
                self.net[layer_num].apply(layer_factory(layer_name, w0=w0)[1])
        elif isinstance(nonlinearity, str):
            if self.weight_init is not None:
                self.net.apply(self.weight_init)

            if self.first_layer_init is not None:
                self.net[0].apply(self.first_layer_init)

        if set_bias is not None:
            self.net[-1][0].bias.data = set_bias * torch.ones_like(self.net[-1][0].bias.data)

    def forward(self, coords):
        output = self.net(coords)
        return output


class RadianceNet(nn.Module):
    def __init__(self, in_features=2, out_features=1,
                 hidden_features=256, num_hidden_layers=3, w0=30,
                 input_pe_params=[('ray_samples', 3, 10), ('ray_orientations', 6, 4)],
                 nl='relu'):

        super().__init__()
        self.input_pe_params = input_pe_params
        self.nl = nl

        self.positional_encoding_fn = {}
        if nl != 'sine':
            for input_to_encode, input_dim, num_pe_fns in self.input_pe_params:
                self.positional_encoding_fn[input_to_encode] = PositionalEncoding(num_encoding_functions=num_pe_fns,
                                                                                  input_dim=input_dim)

        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=nl, w0=w0)

        print(self)

    def forward(self, model_input):

        input_dict = {key: input.clone().detach().requires_grad_(True)
                      for key, input in model_input.items()}

        if self.nl != 'sine':
            for input_to_encode, input_dim, num_pe_fns in self.input_pe_params:
                encoded_input = self.positional_encoding_fn[input_to_encode](input_dict[input_to_encode])
                input_dict.update({input_to_encode: encoded_input})

        input_list = []
        for input_name, _, _ in self.input_pe_params:
            input_list.append(input_dict[input_name])

        coords = torch.cat(input_list, dim=-1)

        if coords.ndim == 2:
            coords = coords[None, :, :]

        output = self.net(coords)

        output_dict = {'output': output}
        return {'model_in': input_dict, 'model_out': output_dict}
