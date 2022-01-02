import torch
import forward_models


def function_mse(model_output, gt):
    idx = model_output['model_in']['idx'].long().squeeze()
    loss = (model_output['model_out']['output'][:, idx] - gt['func'][:, idx]) ** 2
    return {'func_loss': loss.mean()}


def image_mse(model_output, gt):
    if 'complex' in model_output['model_out']:
        c = model_output['model_out']['complex']
        loss = (c.real - gt['img']) ** 2
        imag_loss = (c.imag) ** 2
        return {'func_loss': loss.mean(), 'imag_loss': imag_loss.mean()}

    else:
        loss = (model_output['model_out']['output'] - gt['img']) ** 2
        return {'func_loss': loss.mean()}


def multiscale_image_mse(model_output, gt, use_resized=False):
    if use_resized:
        loss = [(out - gt_img)**2 for out, gt_img in zip(model_output['model_out']['output'], gt['img'])]
    else:
        loss = [(out - gt['img'])**2 for out in model_output['model_out']['output']]

    loss = torch.stack(loss).mean()

    return {'func_loss': loss}


def multiscale_radiance_loss(model_outputs, gt, use_resized=False, weight=1.0,
                             regularize_sigma=False, reg_lambda=1e-5, reg_c=0.5):
    tomo_loss = None
    sigma_reg = None

    pred_sigmas = [pred[..., -1:] for pred in model_outputs['combined']['model_out']['output']]
    pred_rgbs = [pred[..., :-1] for pred in model_outputs['combined']['model_out']['output']]
    if isinstance(model_outputs['combined']['model_in']['t_intervals'], list):
        t_intervals = [t_interval for t_interval in model_outputs['combined']['model_in']['t_intervals']]
    else:
        t_intervals = model_outputs['combined']['model_in']['t_intervals']

    for idx, (pred_sigma, pred_rgb) in enumerate(zip(pred_sigmas, pred_rgbs)):

        if isinstance(t_intervals, list):
            t_interval = t_intervals[idx]
        else:
            t_interval = t_intervals

        # Pass through the forward models
        pred_weights = forward_models.compute_transmittance_weights(pred_sigma, t_interval)
        pred_pixel_samples = forward_models.compute_tomo_radiance(pred_weights, pred_rgb)

        # Target Ground truth
        if use_resized:
            target_pixel_samples = gt['pixel_samples'][idx]
        else:
            target_pixel_samples = gt['pixel_samples']

        # Loss
        if tomo_loss is None:
            tomo_loss = (pred_pixel_samples - target_pixel_samples)**2
        else:
            tomo_loss += (pred_pixel_samples - target_pixel_samples)**2

        if regularize_sigma:
            tau = torch.nn.functional.softplus(pred_sigma - 1)
            if sigma_reg is None:
                sigma_reg = (torch.log(1 + tau**2 / reg_c))
            else:
                sigma_reg += (torch.log(1 + tau**2 / reg_c))

    loss = {'tomo_rad_loss': weight * tomo_loss.mean()}

    if regularize_sigma:
        loss['sigma_reg'] = reg_lambda * sigma_reg.mean()

    return loss


def radiance_sigma_rgb_loss(model_outputs, gt, regularize_sigma=False,
                            reg_lambda=1e-5, reg_c=0.5):
    pred_sigma = model_outputs['combined']['model_out']['output'][..., -1:]
    pred_rgb = model_outputs['combined']['model_out']['output'][..., :-1]
    t_intervals = model_outputs['combined']['model_in']['t_intervals']

    # Pass through the forward models
    pred_weights = forward_models.compute_transmittance_weights(pred_sigma, t_intervals)
    pred_pixel_samples = forward_models.compute_tomo_radiance(pred_weights, pred_rgb)

    # Target Ground truth
    target_pixel_samples = gt['pixel_samples'][..., :3]  # rgba -> rgb

    # Loss
    tomo_loss = (pred_pixel_samples - target_pixel_samples)**2

    if regularize_sigma:
        tau = torch.nn.functional.softplus(pred_sigma - 1)
        sigma_reg = (torch.log(1 + tau**2 / reg_c))

    loss = {'tomo_rad_loss': tomo_loss.mean()}

    if regularize_sigma:
        loss['sigma_reg'] = reg_lambda * sigma_reg.mean()

    return loss


def overfit_sdf(model_output, gt, coarse_loss_weight=1e-2):
    return overfit_sdf_loss_total(model_output, gt, is_multiscale=False,
                                  coarse_loss_weight=coarse_loss_weight)


def multiscale_overfit_sdf(model_output, gt, coarse_loss_weight=1e-2):
    return overfit_sdf_loss_total(model_output, gt, is_multiscale=True,
                                  coarse_loss_weight=coarse_loss_weight)


def overfit_sdf_loss_total(model_output, gt, is_multiscale, lambda_grad=1e-3,
                           coarse_loss_weight=1e-2):
    ''' fit sdf to sphere via mse loss '''

    gt_sdf = gt['sdf']
    pred_sdf = model_output['model_out']

    pred_sdf_ = pred_sdf[0] if is_multiscale else pred_sdf

    mse_ = (gt_sdf - pred_sdf_)**2

    if is_multiscale:
        for pred_sdf_ in pred_sdf[1:]:
            mse_ += (gt_sdf - pred_sdf_)**2

    mse_ = (mse_ / len(pred_sdf))
    mse_[:, ::2] *= coarse_loss_weight

    mse_fine = mse_[:, 1::2].sum()
    mse_coarse = mse_[:, ::2].sum()

    return {'sdf_fine': mse_fine, 'sdf_coarse': mse_coarse}
