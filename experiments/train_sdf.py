import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from torch.utils.data import DataLoader
import configargparse
import dataio
import utils
import training
import loss_functions
import modules
from functools import partial

torch.set_num_threads(8)

p = configargparse.ArgumentParser()

# config file, output directories
p.add('-c', '--config', required=False, is_config_file=True,
      help='Path to config file.')
p.add_argument('--logging_root', type=str, default='../logs',
               help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='subdirectory in logging_root for checkpoints, summaries')

# general training
p.add_argument('--model_type', type=str, default='mfn',
               help='options: mfn, siren, ff')
p.add_argument('--hidden_size', type=int, default=128,
               help='size of hidden layer')
p.add_argument('--hidden_layers', type=int, default=8)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate')
p.add_argument('--num_steps', type=int, default=20000,
               help='number of training steps')
p.add_argument('--ckpt_step', type=int, default=0,
               help='step at which to resume training')
p.add_argument('--gpu', type=int, default=1, help='GPU ID to use')
p.add_argument('--seed', default=None,
               help='random seed for experiment reproducibility')

# mfn options
p.add_argument('--multiscale', action='store_true', default=False,
               help='use multiscale')
p.add_argument('--max_freq', type=int, default=512,
               help='The network-equivalent sample rate used to represent the signal.'
               + 'Should be at least twice the Nyquist frequency.')
p.add_argument('--input_scales', nargs='*', type=float, default=None,
               help='fraction of resolution growth at each layer')
p.add_argument('--output_layers', nargs='*', type=int, default=None,
               help='layer indices to output, beginning at 1')

# mlp options
p.add_argument('--w0', default=30, type=int,
               help='omega_0 parameter for siren')
p.add_argument('--pe_scale', default=5, type=float,
               help='positional encoding scale')

# sdf model and sampling
p.add_argument('--num_pts_on', type=int, default=10000,
               help='number of on-surface points to sample')
p.add_argument('--coarse_scale', type=float, default=1e-1,
               help='laplacian scale factor for coarse samples')
p.add_argument('--fine_scale', type=float, default=1e-3,
               help='laplacian scale factor for fine samples')
p.add_argument('--coarse_weight', type=float, default=1e-2,
               help='weight to apply to coarse loss samples')

# data i/o
p.add_argument('--shape', type=str, default='bunny',
               help='name of point cloud shape in xyz format')
p.add_argument('--point_cloud_path', type=str,
               default='../data/armadillo.xyz',
               help='path for input point cloud')
p.add_argument('--num_workers', default=0, type=int,
               help='number of workers')

# tensorboard summary
p.add_argument('--steps_til_ckpt', type=int, default=50000,
               help='epoch frequency to save a checkpoint')
p.add_argument('--steps_til_summary', type=int, default=1000,
               help='epoch frequency to update tensorboard summary')

opt = p.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)


def main():

    print('--- Run Configuration ---')
    for k, v in vars(opt).items():
        print(k, v)

    train()


def train():

    opt.root_path = os.path.join(opt.logging_root, opt.experiment_name)
    utils.cond_mkdir(opt.root_path)

    if opt.seed:
        torch.manual_seed(int(opt.seed))
        np.random.seed(int(opt.seed))

    dataloader = init_dataloader(opt)

    model = init_model(opt)

    loss_fn, summary_fn = init_loss(opt)

    save_params(opt, model)

    training.train(model=model, train_dataloader=dataloader, steps=opt.num_steps,
                   lr=opt.lr, steps_til_summary=opt.steps_til_summary,
                   ckpt_step=opt.ckpt_step,
                   steps_til_checkpoint=opt.steps_til_ckpt,
                   model_dir=opt.root_path, loss_fn=loss_fn, summary_fn=summary_fn,
                   double_precision=False, clip_grad=True,
                   use_lr_scheduler=True)


def init_dataloader(opt):
    ''' load sdf dataloader via eikonal equation or fitting sdf directly '''

    sdf_dataset = dataio.MeshSDF(opt.point_cloud_path,
                                 num_samples=opt.num_pts_on,
                                 coarse_scale=opt.coarse_scale,
                                 fine_scale=opt.fine_scale)

    dataloader = DataLoader(sdf_dataset, shuffle=True,
                            batch_size=1, pin_memory=True,
                            num_workers=opt.num_workers)

    return dataloader


def init_model(opt):
    ''' return appropriate model given experiment configs '''

    if opt.model_type == 'mfn':

        opt.input_scales = [1/24, 1/24, 1/24, 1/16, 1/16, 1/8, 1/8, 1/4, 1/4]
        opt.output_layers = [2, 4, 6, 8]

        frequency = (opt.max_freq, opt.max_freq, opt.max_freq)

        if opt.multiscale:
            if opt.output_layers and len(opt.output_layers) == 1:
                raise ValueError('expects >1 layer extraction if multiscale')
            model_ = modules.MultiscaleBACON
        else:
            model_ = modules.BACON

        model = model_(in_size=3, hidden_size=opt.hidden_size, out_size=1,
                       hidden_layers=opt.hidden_layers,
                       bias=True,
                       frequency=frequency,
                       quantization_interval=2*np.pi,  # data on range [-0.5, 0.5]
                       input_scales=opt.input_scales,
                       is_sdf=True,
                       output_layers=opt.output_layers,
                       reuse_filters=True)

    elif opt.model_type == 'siren':

        if opt.multiscale:
            model_ = modules.MultiscaleCoordinateNet
        else:
            model_ = modules.CoordinateNet

        model = model_(nl='sine',
                       in_features=3,
                       out_features=1,
                       num_hidden_layers=opt.hidden_layers,
                       hidden_features=opt.hidden_size,
                       w0=opt.w0,
                       is_sdf=True)

    elif opt.model_type == 'ff':  # mlp w relu + positional encoding
        model = modules.CoordinateNet(nl='relu',
                                      in_features=3,
                                      out_features=1,
                                      num_hidden_layers=opt.hidden_layers,
                                      hidden_features=opt.hidden_size,
                                      is_sdf=True,
                                      pe_scale=opt.pe_scale,
                                      use_sigmoid=False)
    else:
        raise NotImplementedError

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Num. Parameters: {params}')
    model.cuda()

    # if resuming model training
    if opt.ckpt_step:
        opt.num_steps -= opt.ckpt_step  # steps remaning to train
        if opt.num_steps < 1:
            raise ValueError('ckpt_epoch must be less than num_epochs')
        print(opt.num_steps)

        pth_file = '{}/checkpoints/model_step_{}.pth'.format(opt.root_path,
                                                             str(opt.ckpt_step).zfill(4))
        model.load_state_dict(torch.load(pth_file))

    return model


def init_loss(opt):
    ''' define loss, summary functions given expmt configs '''

    if opt.multiscale:
        summary_fn = utils.write_multiscale_sdf_summary
        loss_fn = partial(loss_functions.multiscale_overfit_sdf,
                          coarse_loss_weight=opt.coarse_weight)
    else:
        summary_fn = utils.write_sdf_summary
        loss_fn = partial(loss_functions.overfit_sdf,
                          coarse_loss_weight=opt.coarse_weight)

    return loss_fn, summary_fn


def save_params(opt, model):

    # Save command-line parameters log directory.
    p.write_config_file(opt, [os.path.join(opt.root_path, 'config.ini')])
    with open(os.path.join(opt.root_path, "params.txt"), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

    # Save text summary of model into log directory.
    with open(os.path.join(opt.root_path, "model.txt"), "w") as out_file:
        out_file.write(str(model))


if __name__ == '__main__':
    main()
