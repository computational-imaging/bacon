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
from functools import partial
import torch
import numpy as np

torch.set_num_threads(8)
torch.backends.cudnn.benchmark = True

p = configargparse.ArgumentParser()
p.add('-c', '--config', required=False, is_config_file=True, help='Path to config file.')

# Experiment & I/O general properties
p.add_argument('--experiment_name', type=str, default=None,
               help='path to directory where checkpoints & tensorboard events will be saved.')
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--logging_root', type=str, default='../logs', help='root for logging')
p.add_argument('--dataset_path', type=str, default='../data/nerf_synthetic/lego/',
               help='path to directory where dataset is stored')
p.add_argument('--resume', nargs=2, type=str, default=None,
               help='resume training, specify path to directory where model is stored.')
p.add_argument('--num_steps', type=int, default=1000000,
               help='Number of iterations to train for.')
p.add_argument('--steps_til_ckpt', type=int, default=50000,
               help='Iterations until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=2000,
               help='Iterations until tensorboard summary is saved.')

# GPU & other computing properties
p.add_argument('--gpu', type=int, default=0,
               help='GPU ID to use')
p.add_argument('--chunk_size_train', type=int, default=1024,
               help='max chunk size to process data during training')
p.add_argument('--chunk_size_eval', type=int, default=512,
               help='max chunk size to process data during eval')
p.add_argument('--num_workers', type=int, default=0, help='number of dataloader workers.')

# Learning properties
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument('--batch_size', type=int, default=1)

# Network architecture properties
p.add_argument('--hidden_features', type=int, default=128)
p.add_argument('--hidden_layers', type=int, default=4)

p.add_argument('--multiscale', action='store_true', help='use multiscale architecture')
p.add_argument('--supervise_hr', action='store_true', help='supervise only with high resolution signal')
p.add_argument('--use_resized', action='store_true', help='use explicit multiscale supervision')
p.add_argument('--reuse_filters', action='store_true', help='reuse fourier filters for faster training/inference')

# NeRF Properties
p.add_argument('--img_size', type=int, default=64,
               help='image resolution to train on (assumed symmetric)')
p.add_argument('--samples_per_ray', type=int, default=128,
               help='samples to evaluate along each ray')
p.add_argument('--samples_per_view', type=int, default=1024,
               help='samples to evaluate along each view')

opt = p.parse_args()

if opt.experiment_name is None and opt.render_model is None:
    p.error('--experiment_name is required.')

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)


def main():
    print('--- Run Configuration ---')
    for k, v in vars(opt).items():
        print(k, v)
    train()


def train(validation=True):
    root_path = os.path.join(opt.logging_root, opt.experiment_name)
    utils.cond_mkdir(root_path)

    ''' Training dataset '''
    dataset = dataio.NerfBlenderDataset(opt.dataset_path,
                                        splits=['train'],
                                        mode='train',
                                        resize_to=2*(opt.img_size,),
                                        multiscale=opt.multiscale)

    coords_dataset = dataio.Implicit6DMultiviewDataWrapper(dataset,
                                                           dataset.get_img_shape(),
                                                           dataset.get_camera_params(),
                                                           samples_per_view=opt.samples_per_view,
                                                           num_workers=opt.num_workers,
                                                           multiscale=opt.use_resized,
                                                           supervise_hr=opt.supervise_hr,
                                                           scales=[1/8, 1/4, 1/2, 1])
    ''' Validation dataset '''
    if validation:
        val_dataset = dataio.NerfBlenderDataset(opt.dataset_path,
                                                splits=['val'],
                                                mode='val',
                                                resize_to=2*(opt.img_size,),
                                                multiscale=opt.multiscale)

        val_coords_dataset = dataio.Implicit6DMultiviewDataWrapper(val_dataset,
                                                                   val_dataset.get_img_shape(),
                                                                   val_dataset.get_camera_params(),
                                                                   samples_per_view=opt.img_size**2,
                                                                   num_workers=opt.num_workers,
                                                                   multiscale=opt.use_resized,
                                                                   supervise_hr=opt.supervise_hr,
                                                                   scales=[1/8, 1/4, 1/2, 1])

    ''' Dataloaders'''
    dataloader = DataLoader(coords_dataset, shuffle=True, batch_size=opt.batch_size,  # num of views in a batch
                            pin_memory=True, num_workers=opt.num_workers)

    if validation:
        val_dataloader = DataLoader(val_coords_dataset, shuffle=True, batch_size=1,
                                    pin_memory=True, num_workers=opt.num_workers)
    else:
        val_dataloader = None

    # get model paths
    if opt.resume is not None:
        path, step = opt.resume
        step = int(step)
        assert(os.path.isdir(path))
        assert opt.config is not None, 'Specify config file'

    # since model goes between -4 and 4 instead of -0.5 to 0.5
    # we divide by a factor of 8. Then this is Nyquist sampled
    # assuming a maximum frequency of opt.img_size/8 cycles per unit interval
    # (where the blender dataset scenes typically span from -4 to 4 units)
    rgb_sample_freq = 3*(2*opt.img_size/8,)

    if opt.multiscale:
        # scale the frequencies of each layer
        # so that we have outputs at 1/8, 1/4, 1/2, and 1x
        # the maximum network bnadiwdth
        input_scales = [1/24, 1/24, 1/24, 1/16, 1/16, 1/8, 1/8, 1/4, 1/4]
        output_layers = [2, 4, 6, 8]

        model = modules.MultiscaleBACON(3, opt.hidden_features, 4,
                                        hidden_layers=opt.hidden_layers,
                                        bias=True,
                                        frequency=rgb_sample_freq,
                                        quantization_interval=np.pi/4,
                                        input_scales=input_scales,
                                        output_layers=output_layers,
                                        reuse_filters=opt.reuse_filters)
        model.cuda()

    else:
        input_scales = [1/24, 1/24, 1/24, 1/16, 1/16, 1/8, 1/8, 1/4, 1/4]
        input_scales = input_scales[:opt.hidden_layers+1]

        model = modules.BACON(3, opt.hidden_features, 4,
                              hidden_layers=opt.hidden_layers,
                              bias=True,
                              frequency=rgb_sample_freq,
                              quantization_interval=np.pi/4,
                              reuse_filters=opt.reuse_filters,
                              input_scales=input_scales)
        model.cuda()

    if opt.resume is not None:
        print('Loading checkpoints')

        state_dict = torch.load(path + '/checkpoints/' + f'model_combined_step_{step:04d}.pth')
        model.load_state_dict(state_dict, strict=False)

        # load optimizers
        try:
            resume_checkpoint = {}
            ckpt = torch.load(path + '/checkpoints/' + f'optim_combined_step_{step:04d}.pth')
            for g in ckpt['optimizer_state_dict']['param_groups']:
                g['lr'] = opt.lr

            resume_checkpoint['combined'] = {}
            resume_checkpoint['combined']['optim'] = ckpt['optimizer_state_dict']
            resume_checkpoint['combined']['scheduler'] = ckpt['scheduler_state_dict']
            resume_checkpoint['step'] = ckpt['step']
        except FileNotFoundError:
            print('Unable to load optimizer checkpoints')
    else:
        resume_checkpoint = {}

    models = {'combined': model}

    # Define the loss
    if opt.multiscale:
        loss_fn = partial(loss_functions.multiscale_radiance_loss, use_resized=opt.use_resized)
        summary_fn = partial(utils.write_multiscale_radiance_summary,
                             chunk_size_eval=opt.chunk_size_eval,
                             num_views_to_disp_at_training=1,
                             hierarchical_sampling=True)
    else:
        loss_fn = partial(loss_functions.radiance_sigma_rgb_loss)

        summary_fn = partial(utils.write_radiance_summary,
                             chunk_size_eval=opt.chunk_size_eval,
                             num_views_to_disp_at_training=1,
                             hierarchical_sampling=True)

    chunk_lists_from_batch_fn = dataio.chunk_lists_from_batch_reduce_to_raysamples_fn

    # Save command-line parameters log directory.
    p.write_config_file(opt, [os.path.join(root_path, 'config.ini')])
    with open(os.path.join(root_path, "params.txt"), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

    # Save text summary of model into log directory.
    with open(os.path.join(root_path, "model.txt"), "w") as out_file:
        for model_name, model in models.items():
            out_file.write(model_name)
            out_file.write(str(model))

    training.train_wchunks(models, dataloader,
                           num_steps=opt.num_steps, lr=opt.lr,
                           steps_til_summary=opt.steps_til_summary, steps_til_checkpoint=opt.steps_til_ckpt,
                           model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn,
                           val_dataloader=val_dataloader,
                           chunk_lists_from_batch_fn=chunk_lists_from_batch_fn,
                           max_chunk_size=opt.chunk_size_train,
                           resume_checkpoint=resume_checkpoint,
                           chunked=True,
                           hierarchical_sampling=True,
                           stop_after=0)


if __name__ == '__main__':
    main()
