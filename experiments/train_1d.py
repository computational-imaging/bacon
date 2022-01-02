# Enable import from parent package
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.tensorboard import SummaryWriter
import dataio
import utils
import training
import loss_functions
import modules
from torch.utils.data import DataLoader
import configargparse
import torch
from functools import partial
import numpy as np

torch.backends.cudnn.benchmark = True
torch.set_num_threads(4)

p = configargparse.ArgumentParser()
p.add('-c', '--config', required=False, is_config_file=True, help='Path to config file.')

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--hidden_features', type=int, default=128)
p.add_argument('--hidden_layers', type=int, default=4)
p.add_argument('--experiment_name', type=str, default='train_1d_mfn',
               help='path to directory where checkpoints & tensorboard events will be saved.')
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument('--num_steps', type=int, default=1001,
               help='Number of epochs to train for.')
p.add_argument('--gpu', type=int, default=0,
               help='GPU ID to use')

p.add_argument('--model', default='mfn', choices=['mfn', 'mlp'],
               help='use MFN or standard MLP')
p.add_argument('--activation', type=str, default='sine',
               choices=['sine', 'relu', 'requ', 'gelu', 'selu', 'softplus', 'tanh', 'swish'],
               help='activation to use (for model mlp only)')
p.add_argument('--w0', type=float, default=10)
p.add_argument('--pe_scale', type=float, default=3, help='positional encoding scale')
p.add_argument('--no_pe', action='store_true', default=False, help='override to have no positional encoding for relu mlp')
p.add_argument('--max_freq', type=float, default=5, help='The network-equivalent sample rate used to represent the signal. Should be at least twice the Nyquist frequency.')

# summary options
p.add_argument('--steps_til_ckpt', type=int, default=100,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')

# logging options
p.add_argument('--logging_root', type=str, default='../logs', help='root for logging')

opt = p.parse_args()

if opt.experiment_name is None and opt.render_model is None:
    p.error('--experiment_name is required.')

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)


def main():

    print('--- Run Configuration ---')
    for k, v in vars(opt).items():
        print(k, v)

    train()


def train():
    root_path = os.path.join(opt.logging_root, opt.experiment_name)
    utils.cond_mkdir(root_path)

    fn = dataio.sines1
    train_dataset = dataio.Func1DWrapper(range=(-0.5, 0.5),
                                         fn=fn,
                                         sampling_density=1000,
                                         train_every=1000/18)  # 18 samples is ~1.1 the nyquist rate assuming fmax=8

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

    if opt.model == 'mlp':
        model = modules.CoordinateNet(nl=opt.activation,
                                      in_features=1,
                                      out_features=1,
                                      hidden_features=opt.hidden_features,
                                      num_hidden_layers=opt.hidden_layers,
                                      w0=opt.w0,
                                      pe_scale=opt.pe_scale,
                                      use_sigmoid=False,
                                      no_pe=opt.no_pe)

    elif opt.model == 'mfn':
        model = modules.BACON(1, opt.hidden_features, 1,
                              hidden_layers=opt.hidden_layers,
                              bias=True,
                              frequency=[opt.max_freq, ],
                              quantization_interval=2*np.pi)
    else:
        raise ValueError('model must be mlp or mfn')

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Num. Parameters: {params}')

    model.cuda()

    # Define the loss
    loss_fn = loss_functions.function_mse
    summary_fn = partial(utils.write_simple_1D_function_summary, train_dataset)

    # Save command-line parameters log directory.
    p.write_config_file(opt, [os.path.join(root_path, 'config.ini')])
    with open(os.path.join(root_path, "params.txt"), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

    # Save text summary of model into log directory.
    with open(os.path.join(root_path, "model.txt"), "w") as out_file:
        out_file.write(str(model))

    training.train(model=model, train_dataloader=train_dataloader, steps=opt.num_steps, lr=opt.lr,
                   steps_til_summary=opt.steps_til_summary, steps_til_checkpoint=opt.steps_til_ckpt,
                   model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn)


if __name__ == '__main__':
    main()
