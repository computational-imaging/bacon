import torch


def cumprod_exclusive(tensor, dim=-2):
    cumprod = torch.cumprod(tensor, dim)
    cumprod = torch.roll(cumprod, 1, dim)
    cumprod[..., 0, :] = 1.0
    return cumprod


def compute_transmittance_weights(pred_sigma, t_intervals):
    # pred_alpha = 1.-torch.exp(-torch.relu(pred_sigma)*t_intervals)
    tau = torch.nn.functional.softplus(pred_sigma - 1)
    pred_alpha = 1.-torch.exp(-tau*t_intervals)
    pred_weights = pred_alpha * cumprod_exclusive(1.-pred_alpha+1e-10, dim=-2)
    return pred_weights


def compute_tomo_radiance(pred_weights, pred_rgb, black_background=False):
    eps = 0.001
    pred_rgb_pos = torch.sigmoid(pred_rgb)
    pred_rgb_pos = pred_rgb_pos * (1 + 2 * eps) - eps
    pred_pixel_samples = torch.sum(pred_rgb_pos*pred_weights, dim=-2)  # line integral

    if not black_background:
        pred_pixel_samples += 1 - pred_weights.sum(-2)
    return pred_pixel_samples


def compute_tomo_depth(pred_weights, zs):
    pred_depth = torch.sum(pred_weights*zs, dim=-2)
    return pred_depth


def compute_disp_from_depth(pred_depth, pred_weights):
    pred_disp = 1. / torch.max(torch.tensor(1e-10).to(pred_depth.device),
                               pred_depth / torch.sum(pred_weights, -2))
    return pred_disp
