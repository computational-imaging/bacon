import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import modules
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import warnings
import figure_setup


def make_square_axes(ax, scale=1):
    """Make an axes square in screen units.
    Should be called after plotting.
    """
    ax.set_aspect(scale / ax.get_data_ratio())


# original MFN implementation
class MFNBase(nn.Module):
    """
    Multiplicative filter network base class.
    Expects the child class to define the 'filters' attribute, which should be
    a nn.ModuleList of n_layers+1 filters with output equal to hidden_size.
    """

    def __init__(
        self, hidden_size, out_size, n_layers, weight_scale, bias=True, output_act=False
    ):
        super().__init__()

        self.linear = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias) for _ in range(n_layers)]
        )
        self.output_linear = nn.Linear(hidden_size, out_size)
        self.output_act = output_act

        for lin in self.linear:
            lin.weight.data.uniform_(
                -np.sqrt(weight_scale / hidden_size),
                np.sqrt(weight_scale / hidden_size),
            )

        return

    def forward(self, x):
        out = self.filters[0](x)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.linear[i - 1](out)
        out = self.output_linear(out)

        if self.output_act:
            out = torch.sin(out)

        return out


class FourierLayer(nn.Module):
    """
    Sine filter as used in FourierNet.
    """

    def __init__(self, in_features, out_features, weight_scale):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.data *= weight_scale  # gamma
        self.linear.bias.data.uniform_(-np.pi, np.pi)
        return

    def forward(self, x):
        return torch.sin(self.linear(x))


class FourierNet(MFNBase):
    def __init__(
        self,
        in_size,
        hidden_size,
        out_size,
        n_layers=3,
        input_scale=256.0,
        weight_scale=1.0,
        bias=True,
        output_act=False,
    ):
        super().__init__(
            hidden_size, out_size, n_layers, weight_scale, bias, output_act
        )
        self.filters = nn.ModuleList(
            [
                FourierLayer(in_size, hidden_size, input_scale / np.sqrt(n_layers + 1))
                for _ in range(n_layers + 1)
            ]
        )


def plot_initialization(use_original=False, plot_fit=True):

    f = 256
    nl = 8

    if use_original:
        model = FourierNet(1, 1024, 1, nl)
    else:
        model = modules.BACON(1, 1024, 1, nl, frequency=(f, f))

    coords = torch.linspace(-0.5, 0.5, 1000)[:, None]

    activations = []
    activations.append(coords.mean(-1))
    out = model.filters[0].linear(coords)
    activations.append(out.flatten())
    out = model.filters[0](coords)
    activations.append(out.flatten())
    for i in range(1, len(model.filters)):
        activations.append(model.linear[i-1](out).flatten())
        out = model.filters[i](coords) * model.linear[i - 1](out)
        activations.append(out.flatten())

    figure_setup.figure_setup()
    fig = plt.figure(figsize=(6.9, 5))
    gs = fig.add_gridspec(5, 4, wspace=0.25, hspace=0.25)
    gs.update(left=0.05, right=1.0, top=0.95, bottom=0.05)

    for idx, act in enumerate(activations):

        if idx > 2:
            # plt.subplot(2, len(activations)//2 + 1, idx+2)
            fig.add_subplot(gs[(idx+1)//4, (idx+1) % 4])
        else:
            fig.add_subplot(gs[0, idx])

            # plt.subplot(2, len(activations)//2 + 1, idx+1)
        plt.hist(act.detach().cpu().numpy(), 50, density=True)

        if idx == 0:
            plt.title('Input')
            plt.xlim(-0.5, 0.5)

        if idx == 1:
            x = np.linspace(-120, 120, 1000)

            if plot_fit:
                plt.plot(x, 2/(2*np.pi*f/(nl+1)) * np.log(np.pi*f/(nl+1)/(np.minimum(abs(2*x), np.pi*f/(nl+1)))), 'r')
            plt.xlim(-120, 120)
            plt.title('Before Sine')

        if idx == 2:
            x = np.linspace(-1, 1, 1000)

            if plot_fit:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    plt.plot(x, 1/np.pi * 1 / (np.sqrt(1 - x**2)), 'r')
            plt.xlim(-1, 1)
            plt.title('After Sine')

        if idx % 2 == 1 and idx > 2:
            x = np.linspace(-4, 4, 1000)
            if plot_fit:
                plt.plot(x, 1 / np.sqrt(2*np.pi) * np.exp(-x**2/2), 'r')
            plt.xlim(-4, 4)
            plt.title('After Linear')

        if idx % 2 == 0 and idx > 2:
            # this is the product of a standard normal and
            # an arcsine distributed RV, which is an RV
            # that obeys the product rule
            # https://en.wikipedia.org/wiki/Distribution_of_the_product_of_two_random_variables
            # and its variance will be 1/2. Then, we can use the initialization scheme of siren (Thm 1.8)

            zs = np.linspace(-4, 4, 1000)
            x = np.linspace(-4, 4, 5000)
            out = np.zeros_like(zs)
            dx = x[1] - x[0]
            for idx, z in enumerate(zs):
                zdx2 = (z/x)**2
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    tmp = 1 / np.sqrt(2*np.pi) * np.exp(-x**2/2) * 1/np.pi * 1/np.sqrt(1 - zdx2) * 1/abs(x) * dx
                tmp[np.isnan(tmp)] = 0
                out[idx] = tmp.sum()

            if plot_fit:
                plt.plot(zs, out, 'r')
            plt.title('After Product')
            plt.xlim(-4, 4)

        make_square_axes(plt.gca(), 0.5)
    plt.show()


if __name__ == '__main__':
    plot_initialization(use_original=False, plot_fit=True)
    plot_initialization(use_original=True, plot_fit=False)
