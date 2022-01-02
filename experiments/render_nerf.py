import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
import dataio
import modules
import forward_models
import training
import torch
import numpy as np
import configargparse
import dataclasses
from dataclasses import dataclass
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import skimage.io
from functools import partial


torch.backends.cudnn.benchmark = True

ssim_fn = partial(structural_similarity, data_range=1,
                  gaussian_weights=True, sigma=1.5,
                  use_sample_covariance=False,
                  multichannel=True,
                  channel_axis=-1)


@dataclass
class Options:
    config: str
    experiment_name: str
    logging_root: str
    dataset_path: str
    num_epochs: int
    epochs_til_ckpt: int
    steps_til_summary: int
    gpu: int
    img_size: int
    chunk_size_train: int
    chunk_size_eval: int
    num_workers: int
    lr: float
    batch_size: int
    hidden_features: int
    hidden_layers: int
    model: str
    activation: str
    multiscale: bool
    single_network: bool
    use_resized: bool
    reuse_filters: bool
    samples_per_ray: int
    samples_per_view: int
    forward_mode: str
    supervise_hr: bool
    rank: int

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, self.__annotations__[k](v))

        if 'supervise_hr' not in kwargs.keys():
            self.supervise_hr = False

        self.img_size = 512


def load_dataset(opt, res, scale):
    dataset = dataio.NerfBlenderDataset(opt.dataset_path,
                                        splits=['test'],
                                        mode='test',
                                        resize_to=(int(res/2**(3-scale)), int(res/2**(3-scale))),
                                        multiscale=opt.multiscale,
                                        override_scale=scale,
                                        testskip=1)

    coords_dataset = dataio.Implicit6DMultiviewDataWrapper(dataset,
                                                           (int(res/2**(3-scale)), int(res/2**(3-scale))),
                                                           dataset.get_camera_params(),
                                                           samples_per_ray=256,  # opt.samples_per_ray,
                                                           samples_per_view=opt.samples_per_view,
                                                           num_workers=opt.num_workers,
                                                           multiscale=opt.use_resized,
                                                           supervise_hr=opt.supervise_hr,
                                                           scales=[1/8, 1/4, 1/2, 1])
    coords_dataset.toggle_logging_sampling()

    return coords_dataset


def load_model(opt, checkpoint):
    # since model goes between -4 and 4 instead of -0.5 to 0.5
    sample_frequency = 3*(opt.img_size/4,)

    if opt.multiscale:
        # scale the frequencies of each layer accordingly
        input_scales = [1/24, 1/24, 1/24, 1/16, 1/16, 1/8, 1/8, 1/4, 1/4]
        output_layers = [2, 4, 6, 8]

        with utils.HiddenPrint():
            model = modules.MultiscaleBACON(3, opt.hidden_features, 4,
                                            hidden_layers=opt.hidden_layers,
                                            bias=True,
                                            frequency=sample_frequency,
                                            quantization_interval=np.pi/4,
                                            input_scales=input_scales,
                                            output_layers=output_layers,
                                            reuse_filters=opt.reuse_filters)
        model.cuda()

    print('Loading checkpoints')
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict, strict=False)

    models = {'combined': model}

    return models


def render_in_chunks(in_dict, model, chunk_size, return_all=False):
    batches, rays, samples, dims = in_dict['ray_samples'].shape

    in_dict['ray_samples'] = in_dict['ray_samples'].reshape(-1, 3)

    if return_all:
        out = [torch.zeros(batches, rays, samples, 4, device=in_dict['ray_samples'].device) for i in range(4)]
        out = [o.reshape(-1, 4) for o in out]
    else:
        out = torch.zeros(batches, rays, samples, 4, device=in_dict['ray_samples'].device)
        out = out.reshape(-1, 4)

    chunk_size *= 128
    num_chunks = int(np.ceil(rays*samples / (chunk_size)))

    for i in range(num_chunks):
        tmp = {'ray_samples': in_dict['ray_samples'][i*chunk_size:(i+1)*chunk_size, ...]}

        if return_all:
            for j in range(4):
                out[j][i*chunk_size:(i+1)*chunk_size, ...] = model(tmp)['model_out']['output'][j]
        else:
            out[i*chunk_size:(i+1)*chunk_size, ...] = model(tmp)['model_out']['output'][-1]

    if return_all:
        out = [o.reshape(batches, rays, samples, 4) for o in out]
        return {'model_out': {'output': out}, 'model_in': {'t_intervals': in_dict['t_intervals']}}

    else:
        out = out.reshape(batches, rays, samples, 4)
        return {'model_out': {'output': [out]}, 'model_in': {'t_intervals': in_dict['t_intervals']}}


def render_all_in_chunks(in_dict, model, scale, chunk_size):
    batches, rays, samples, dims = in_dict['ray_samples'].shape
    out = torch.zeros(batches, rays, 3)
    num_chunks = int(np.ceil(rays / (chunk_size)))

    with torch.no_grad():
        for i in range(num_chunks):
            # transfer to cuda
            model_in = {k: v[:, i*chunk_size:(i+1)*chunk_size, ...].cuda() for k, v in in_dict.items()}
            model_in = training.dict2cuda(model_in)

            # run first forward pass for importance sampling
            model.stop_after = 0
            model_out = {'combined': model(model_in)}

            # resample rays
            model_in = training.sample_pdf(model_in, model_out, idx=0)

            # importance sampled pass
            model.stop_after = scale
            model_out = {'combined': model(model_in)}

            # render outputs
            sigma = model_out['combined']['model_out']['output'][-1][..., -1:]
            rgb = model_out['combined']['model_out']['output'][-1][..., :-1]

            t_interval = model_in['t_intervals']

            pred_weights = forward_models.compute_transmittance_weights(sigma, t_interval)
            pred_pixels = forward_models.compute_tomo_radiance(pred_weights, rgb)

            out[:, i*chunk_size:(i+1)*chunk_size, ...] = pred_pixels.cpu()

        pred_view = out.view(int(np.sqrt(rays)), int(np.sqrt(rays)), 3).detach().cpu()
        pred_view = torch.clamp(pred_view, 0, 1).numpy()
    return pred_view


def render_image(opt, models, dataset, chunk_size,
                 in_dict, meta_dict, gt_dict, scale,
                 return_all=False):

    # add batch dimension
    for k, v in in_dict.items():
        in_dict[k].unsqueeze_(0)

    for i in range(len(gt_dict['pixel_samples'])):
        gt_dict['pixel_samples'][i].unsqueeze_(0)

    use_chunks = True
    if in_dict['ray_samples'].shape[1] < chunk_size:
        use_chunks = False
    use_chunks = True

    # render the whole thing in chunks
    if scale > 3:
        pred_view = render_all_in_chunks(in_dict, models['combined'], scale, chunk_size)
        return pred_view, 0.0, 0.0, 0.0

    in_dict = training.dict2cuda(in_dict)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    with torch.no_grad():
        models['combined'].stop_after = 0
        if use_chunks:
            out_dict = {key: render_in_chunks(in_dict, model, chunk_size)
                        for key, model in models.items()}
        else:
            out_dict = {key: model(in_dict) for key, model in models.items()}
        models['combined'].stop_after = scale

        in_dict = training.sample_pdf(in_dict, out_dict, idx=0)

        if use_chunks:
            out_dict = {key: render_in_chunks(in_dict, model, chunk_size, return_all=return_all)
                        for key, model in models.items()}

        else:
            out_dict = {key: model(in_dict) for key, model in models.items()}

        if return_all:
            sigma = [s[..., -1:] for s in out_dict['combined']['model_out']['output']]
            rgb = [c[..., :-1] for c in out_dict['combined']['model_out']['output']]
            t_interval = in_dict['t_intervals']

            if isinstance(gt_dict['pixel_samples'], list):
                gt_view = gt_dict['pixel_samples'][scale].squeeze(0).numpy()
            else:
                gt_view = gt_dict['pixel_samples'].detach().squeeze(0).numpy()
            view_shape = gt_view.shape

            pred_weights = [forward_models.compute_transmittance_weights(s, t_interval) for s in sigma]
            pred_pixels = [forward_models.compute_tomo_radiance(w, c) for w, c in zip(pred_weights, rgb)]

            pred_view = [p.view(view_shape).detach().cpu() for p in pred_pixels]
            pred_view = [torch.clamp(p, 0, 1).numpy() for p in pred_view]

            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)

            return pred_view, 0, 0, elapsed

        else:
            sigma = out_dict['combined']['model_out']['output'][-1][..., -1:]
            rgb = out_dict['combined']['model_out']['output'][-1][..., :-1]
            t_interval = in_dict['t_intervals']

            if isinstance(gt_dict['pixel_samples'], list):
                gt_view = gt_dict['pixel_samples'][scale].squeeze(0).numpy()
            else:
                gt_view = gt_dict['pixel_samples'].detach().squeeze(0).numpy()
            view_shape = gt_view.shape

            pred_weights = forward_models.compute_transmittance_weights(sigma, t_interval)
            pred_pixels = forward_models.compute_tomo_radiance(pred_weights, rgb)

        # log the images
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)

        pred_view = pred_pixels.view(view_shape).detach().cpu()
        pred_view = torch.clamp(pred_view, 0, 1).numpy()

    psnr = peak_signal_noise_ratio(gt_view, pred_view)
    ssim = ssim_fn(gt_view, pred_view)

    return pred_view, psnr, ssim, elapsed


def eval_nerf_bacon(scene, config, checkpoint, outdir, res, scale, chunk_size=10000, return_all=False, val_idx=None):

    os.makedirs(f'./outputs/nerf/{outdir}', exist_ok=True)

    p = configargparse.DefaultConfigFileParser()
    with open(config) as f:
        opt = p.parse(f)

    opt = Options(**opt)
    dataset = load_dataset(opt, res, scale)
    models = load_model(opt, checkpoint)

    for k in models.keys():
        models[k].stop_after = scale

    # render images
    psnrs = []
    ssims = []
    dataset_generator = iter(dataset)
    for idx in range(len(dataset)):

        if val_idx is not None:
            dataset.val_idx = val_idx
            idx = val_idx

        in_dict, meta_dict, gt_dict = next(dataset_generator)

        images, psnr, ssim, elapsed = render_image(opt, models, dataset, chunk_size,
                                                   in_dict, meta_dict, gt_dict,
                                                   scale, return_all=return_all)

        tqdm.write(f'Scale: {scale} | PSNR: {psnr:.02f} dB, SSIM: {ssim:.02f}, Elapsed: {elapsed:.02f} ms')

        if return_all:
            for s in range(4):
                skimage.io.imsave(f'./outputs/nerf/{outdir}/r_{idx}_d{3-s}.png', (images[s]*255).astype(np.uint8))
        else:
            np.save(f'./outputs/nerf/{outdir}/r_{idx}_d{3-scale}.npy', {'psnr': psnr, 'ssim': ssim})
            skimage.io.imsave(f'./outputs/nerf/{outdir}/r_{idx}_d{3-scale}.png', (images*255).astype(np.uint8))

            psnrs.append(psnr)
            ssims.append(ssim)

        if val_idx is not None:
            break

    if not return_all and val_idx is not None:
        np.save(f'./outputs/nerf/{outdir}/metrics_d{3-scale}.npy', {'psnr': psnrs, 'ssim': ssims,
                                                                    'avg_psnr': np.mean(psnrs),
                                                                    'avg_ssim': np.mean(ssims)})

        print(f'Avg. PSNR: {np.mean(psnrs):.02f}, Avg. SSIM: {np.mean(ssims):.02f}')


if __name__ == '__main__':
    # before running this you need to download the nerf blender datasets for the lego model
    # and place in ../data/nerf_synthetic/lego
    # these can be downloaded here
    # https://drive.google.com/drive/folders/1lrDkQanWtTznf48FCaW5lX9ToRdNDF1a

    # render the model trained with explicit supervision at each scale
    config = './config/nerf/bacon.ini'
    checkpoint = '../trained_models/lego.pth'
    outdir = 'lego'
    res = 512
    for scale in range(4):
        eval_nerf_bacon('lego', config, checkpoint, outdir, res, scale)

    # render the semisupervised model
    config = './config/nerf/bacon_semisupervise.ini'
    checkpoint = '../trained_models/lego_semisupervise.pth'
    outdir = 'lego_semisupervise'
    res = 512
    for scale in range(4):
        eval_nerf_bacon('lego_semisupervise', config, checkpoint, outdir, res, scale)
