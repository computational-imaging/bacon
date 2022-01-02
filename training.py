from torch.utils.tensorboard import SummaryWriter
import torch
import utils
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import forward_models
from functools import partial
import shutil


def train(model, train_dataloader, steps, lr, steps_til_summary,
          steps_til_checkpoint, model_dir, loss_fn, summary_fn,
          prefix_model_dir='', val_dataloader=None, double_precision=False,
          clip_grad=False, use_lbfgs=False, loss_schedules=None, params=None,
          ckpt_step=0, use_lr_scheduler=False):

    if params is None:
        optim = torch.optim.Adam(lr=lr, params=model.parameters(), amsgrad=True)
    else:
        optim = torch.optim.Adam(lr=lr, params=params, amsgrad=True)

    if use_lbfgs:
        optim = torch.optim.LBFGS(lr=lr, params=model.parameters(),
                                  max_iter=50000, max_eval=50000,
                                  history_size=50, line_search_fn='strong_wolfe')

    scheduler = None
    if use_lr_scheduler:
        def sampling_scheduler(step, start=0, lr0=1e-4, lrn=1e-4):

            if step > start:
                fine_scale = lr_log_schedule(step-start, num_steps=steps-start, nw=1, lr0=lr0, lrn=lrn)
                train_dataloader.dataset.fine_scale = fine_scale
            else:
                train_dataloader.dataset.fine_scale = lr0

        # lr scheduler
        optim.param_groups[0]['lr'] = 1
        log_scheduler = partial(lr_log_schedule, num_steps=steps, nw=1, lr0=lr, lrn=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=log_scheduler)

    if os.path.exists(model_dir):
        pass
    else:
        os.makedirs(model_dir)

    model_dir_postfixed = os.path.join(model_dir, prefix_model_dir)

    summaries_dir = os.path.join(model_dir_postfixed, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir_postfixed, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    # e.g. epochs=1k, len(train_dataloader)=25
    train_generator = iter(train_dataloader)

    with tqdm(total=steps) as pbar:
        train_losses = []
        for step in range(steps):

            if not step % steps_til_checkpoint and step:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir,
                           'model_step_%04d.pth' % (step + ckpt_step)))
                np.savetxt(os.path.join(checkpoints_dir,
                           'train_losses_step_%04d.txt' % (step + ckpt_step)),
                           np.array(train_losses))

            try:
                # sampling_scheduler(step)
                model_input, gt = next(train_generator)
            except StopIteration:
                train_generator = iter(train_dataloader)
                model_input, gt = next(train_generator)

            start_time = time.time()

            model_input = dict2cuda(model_input)
            gt = dict2cuda(gt)

            if double_precision:
                model_input = {key: value.double()
                               for key, value in model_input.items()}
                gt = {key: value.double() for key, value in gt.items()}

            if use_lbfgs:
                def closure():
                    optim.zero_grad(set_to_none=True)
                    model_output = model(model_input)
                    losses = loss_fn(model_output, gt)
                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        train_loss += loss.mean()
                    train_loss.backward()
                    return train_loss
                optim.step(closure)

            model_output = model(model_input)
            losses = loss_fn(model_output, gt)

            train_loss = 0.
            for loss_name, loss in losses.items():
                single_loss = loss.mean()

                if loss_schedules is not None and \
                        loss_name in loss_schedules:
                    writer.add_scalar(loss_name + "_weight",
                                      loss_schedules[loss_name](step), step)
                    single_loss *= loss_schedules[loss_name](step)

                writer.add_scalar(loss_name, single_loss, step)
                train_loss += single_loss

            train_losses.append(train_loss.item())
            writer.add_scalar("total_train_loss", train_loss, step)
            writer.add_scalar("lr", optim.param_groups[0]['lr'], step)

            if not step % steps_til_summary:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir,
                                        'model_current.pth'))
                summary_fn(model, model_input, gt, model_output, writer, step)

            if not use_lbfgs:
                optim.zero_grad(set_to_none=True)
                train_loss.backward()

                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                optim.step()

                if scheduler is not None:
                    scheduler.step()

            pbar.update(1)

            if not step % steps_til_summary:
                tqdm.write("Step %d, Total loss %0.6f, iteration time %0.6f" % (step, train_loss, time.time() - start_time))

                if val_dataloader is not None:
                    print("Running validation set...")
                    model.eval()
                    with torch.no_grad():
                        val_losses = []
                        for (model_input, gt) in val_dataloader:
                            model_output = model(model_input)
                            val_loss = loss_fn(model_output, gt)
                            val_losses.append(val_loss)

                        writer.add_scalar("val_loss", np.mean(val_losses), step)
                    model.train()

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))


def dict2cuda(a_dict):
    tmp = {}
    for key, value in a_dict.items():
        if isinstance(value, torch.Tensor):
            tmp.update({key: value.cuda()})
        elif isinstance(value, dict):
            tmp.update({key: dict2cuda(value)})
        elif isinstance(value, list) or isinstance(value, tuple):
            if isinstance(value[0], torch.Tensor):
                tmp.update({key: [v.cuda() for v in value]})
        else:
            tmp.update({key: value})
    return tmp


def dict2cpu(a_dict):
    tmp = {}
    for key, value in a_dict.items():
        if isinstance(value, torch.Tensor):
            tmp.update({key: value.cpu()})
        elif isinstance(value, dict):
            tmp.update({key: dict2cpu(value)})
        elif isinstance(value, list):
            if isinstance(value[0], torch.Tensor):
                tmp.update({key: [v.cpu() for v in value]})
        else:
            tmp.update({key: value})
    return tmp


def reg_schedule(it, num_steps=1e6, lr0=1e-3, lrn=1e-4):
    return np.exp((1 - it/num_steps) * np.log(lr0) + (it/num_steps) * np.log(lrn))


def lr_log_schedule(it, num_steps=1e6, nw=2500, lr0=1e-3, lrn=5e-6, lambdaw=0.01):
    return (lambdaw + (1 - lambdaw) * np.sin(np.pi/2 * np.clip(it/nw, 0, 1))) \
        * np.exp((1 - it/num_steps) * np.log(lr0) + (it/num_steps) * np.log(lrn))


def train_wchunks(models, train_dataloader, num_steps, lr, steps_til_summary, steps_til_checkpoint, model_dir,
                  loss_fn, summary_fn, chunk_lists_from_batch_fn,
                  val_dataloader=None, double_precision=False, clip_grad=False, loss_schedules=None,
                  num_cuts=128,
                  max_chunk_size=4096,
                  resume_checkpoint={},
                  chunked=True,
                  hierarchical_sampling=False,
                  coarse_loss_weight=0.1,
                  stop_after=None):

    optims = {key: torch.optim.Adam(lr=1, params=model.parameters())
              for key, model in models.items()}
    schedulers = {key: torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_log_schedule)
                  for key, optim in optims.items()}

    # load optimizer if supplied
    for key in models.keys():
        if key in resume_checkpoint:
            optims[key].load_state_dict(resume_checkpoint[key]['optim'])
            schedulers[key].load_state_dict(resume_checkpoint[key]['scheduler'])

    if os.path.exists(os.path.join(model_dir, 'summaries')):
        val = input("The model directory %s exists. Overwrite? (y/n)" % model_dir)
        if val == 'y':
            if os.path.exists(os.path.join(model_dir, 'summaries')):
                shutil.rmtree(os.path.join(model_dir, 'summaries'))
            if os.path.exists(os.path.join(model_dir, 'checkpoints')):
                shutil.rmtree(os.path.join(model_dir, 'checkpoints'))

    os.makedirs(model_dir, exist_ok=True)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    start_step = 0
    if 'step' in resume_checkpoint:
        start_step = resume_checkpoint['step']

    train_generator = iter(train_dataloader)

    with tqdm(total=num_steps) as pbar:
        pbar.update(start_step)
        train_losses = []
        for step in range(start_step, num_steps):
            if not step % steps_til_checkpoint and step:
                for key, model in models.items():
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_'+key+'_step_%04d.pth' % step))
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_step_%04d.txt' % step),
                               np.array(train_losses))
                for key, optim in optims.items():
                    torch.save({'step': step,
                                'optimizer_state_dict': optim.state_dict(),
                                'scheduler_state_dict': schedulers[key].state_dict()},
                               os.path.join(checkpoints_dir, 'optim_'+key+'_step_%04d.pth' % step))

            try:
                model_input, meta, gt = next(train_generator)
            except StopIteration:
                train_dataloader.dataset.shuffle_rays()
                train_generator = iter(train_dataloader)
                model_input, meta, gt = next(train_generator)

            start_time = time.time()

            for optim in optims.values():
                optim.zero_grad(set_to_none=True)

            batch_avged_losses = {}
            if chunked:
                list_chunked_model_input, list_chunked_meta, list_chunked_gt = \
                    chunk_lists_from_batch_fn(model_input, meta, gt, max_chunk_size)

                num_chunks = len(list_chunked_gt)
                batch_avged_tot_loss = 0.
                for chunk_idx, (chunked_model_input, chunked_meta, chunked_gt) \
                        in enumerate(zip(list_chunked_model_input, list_chunked_meta, list_chunked_gt)):
                    chunked_model_input = dict2cuda(chunked_model_input)
                    chunked_meta = dict2cuda(chunked_meta)
                    chunked_gt = dict2cuda(chunked_gt)

                    # forward pass through model
                    for k in models.keys():
                        models[k].stop_after = stop_after
                    chunk_model_outputs = {key: model(chunked_model_input) for key, model in models.items()}
                    for k in models.keys():
                        models[k].stop_after = None

                    losses = {}

                    if hierarchical_sampling:
                        # set idx to use sigma from coarse level (idx=0)
                        # for hierarchical sampling
                        chunked_model_input_fine = sample_pdf(chunked_model_input,
                                                              chunk_model_outputs, idx=0)

                        chunk_model_importance_outputs = {key: model(chunked_model_input_fine)
                                                          for key, model in models.items()}

                        reg_lambda = reg_schedule(step)
                        losses_importance = loss_fn(chunk_model_importance_outputs, chunked_gt,
                                                    regularize_sigma=True, reg_lambda=reg_lambda)

                    # loss from forward pass
                    train_loss = 0.
                    for loss_name, loss in losses.items():

                        single_loss = loss.mean()
                        train_loss += single_loss / num_chunks

                        batch_avged_tot_loss += float(single_loss / num_chunks)
                        if loss_name in batch_avged_losses:
                            batch_avged_losses[loss_name] += single_loss / num_chunks
                        else:
                            batch_avged_losses.update({loss_name: single_loss/num_chunks})

                    # Loss from eventual second pass
                    if hierarchical_sampling:
                        for loss_name, loss in losses_importance.items():
                            single_loss = loss.mean()
                            train_loss += single_loss / num_chunks

                            batch_avged_tot_loss += float(train_loss)
                            if loss_name + '_importance' in batch_avged_losses:
                                batch_avged_losses[loss_name+'_importance'] += single_loss / num_chunks
                            else:
                                batch_avged_losses.update({loss_name + '_importance': single_loss / num_chunks})

                    train_loss.backward()
            else:
                model_input = dict2cuda(model_input)
                meta = dict2cuda(meta)
                gt = dict2cuda(gt)

                model_outputs = {key: model(model_input) for key, model in models.items()}
                losses = loss_fn(model_outputs, gt)

                # loss from forward pass
                train_loss = 0.
                for loss_name, loss in losses.items():

                    single_loss = loss.mean()
                    train_loss += single_loss

                    batch_avged_tot_loss = float(single_loss)
                    if loss_name in batch_avged_losses:
                        batch_avged_losses[loss_name] += single_loss
                    else:
                        batch_avged_losses.update({loss_name: single_loss})

                train_loss.backward()

            for loss_name, loss in batch_avged_losses.items():
                writer.add_scalar(loss_name, loss, step)
            train_losses.append(batch_avged_tot_loss)
            writer.add_scalar("total_train_loss", batch_avged_tot_loss, step)
            writer.add_scalar("reg_lambda", reg_lambda, step)

            for k in optims.keys():
                writer.add_scalar(f"{k}_lr", optims[k].param_groups[0]['lr'], step)

            if clip_grad:
                for model in models.values():
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

            for optim in optims.values():
                optim.step()

            if not step % steps_til_summary:
                tqdm.write("Step %d, Total loss %0.6f, iteration time %0.6f" % (step, train_loss, time.time() - start_time))
                for key, model in models.items():
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_'+key+'_current.pth'))
                for key, optim in optims.items():
                    torch.save({'step': step,
                                'total_steps': step,
                                'optimizer_state_dict': optim.state_dict(),
                                'scheduler_state_dict': schedulers[key].state_dict()},
                               os.path.join(checkpoints_dir, 'optim_'+key+'_current.pth'))
                summary_fn(models, train_dataloader, val_dataloader, loss_fn, optims, meta, gt,
                           writer, step)

            pbar.update(1)

            for k in schedulers.keys():
                schedulers[k].step()

        for key, model in models.items():
            torch.save(model.state_dict(),
                       os.path.join(checkpoints_dir, 'model_' + key + '_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))


def sample_pdf(model_inputs, model_outputs, offset=5e-3,
               idx=-1):
    ''' hierarchical sampling code for neural radiance fields '''

    z_vals = model_inputs['t']
    bins = .5*(z_vals[..., 1:, :] + z_vals[..., :-1, :]).squeeze()
    bins = bins.clone().detach().requires_grad_(True)

    if 'combined' in model_outputs:
        if isinstance(model_outputs['combined']['model_out']['output'], list):
            pred_sigma = model_outputs['combined']['model_out']['output'][idx][..., -1:]
            t_intervals = model_outputs['combined']['model_in']['t_intervals']
        else:
            pred_sigma = model_outputs['combined']['model_out']['output'][..., -1:]
            t_intervals = model_outputs['combined']['model_in']['t_intervals']
    else:
        pred_sigma = model_outputs['sigma']['model_out']['output']
        t_intervals = model_outputs['sigma']['model_in']['t_intervals']

    if isinstance(pred_sigma, list):
        pred_sigma = pred_sigma[idx]

    pred_weights = forward_models.compute_transmittance_weights(pred_sigma, t_intervals)[..., :-1, 0]

    # blur weights
    pred_weights = torch.cat((pred_weights, pred_weights[..., -1:]), dim=-1)
    weights_max = torch.maximum(pred_weights[..., :-1], pred_weights[..., 1:])
    weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])
    pred_weights = weights_blur + offset

    pdf = pred_weights / torch.sum(pred_weights, dim=-1, keepdim=True)

    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1).squeeze()  # batch_pixels, num_bins=samples_per_ray-1)
    cdf = cdf.detach()
    num_samples = pred_sigma.shape[-2]
    u = torch.rand(list(cdf.shape[:-1])+[num_samples], device=pred_weights.device)

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds), inds-1)
    above = torch.min((cdf.shape[-1]-1)*torch.ones_like(inds), inds)
    inds_g = torch.stack((below, above), -1)

    matched_shape = (inds_g.shape[0], inds_g.shape[1], cdf.shape[-1])
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0])/denom
    t_vals = (bins_g[..., 0] + t*(bins_g[..., 1]-bins_g[..., 0])).unsqueeze(-1)
    t_vals, _ = torch.sort(t_vals, dim=-2)

    ray_dirs = model_inputs['ray_directions']
    ray_orgs = model_inputs['ray_origins']

    t_vals = t_vals[..., 0]
    t_intervals = t_vals[..., 1:] - t_vals[..., :-1]
    t_intervals = torch.cat((t_intervals, 1e10*torch.ones_like(t_intervals[:, 0:1])), dim=-1)
    t_intervals = (t_intervals * ray_dirs.norm(p=2, dim=-1))[..., None]
    t_vals = t_vals[..., None]

    if ray_dirs.ndim == 4:
        t_vals = t_vals[None, ...]

    model_inputs.update({'t': t_vals})
    model_inputs.update({'ray_samples': ray_orgs + ray_dirs * t_vals})
    model_inputs.update({'t_intervals': t_intervals})

    return model_inputs
