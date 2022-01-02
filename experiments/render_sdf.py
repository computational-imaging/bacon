import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
import modules
import torch
import numpy as np
from tqdm import tqdm
import mcubes
import trimesh
import dataio
import math


def export_model(ckpt_path, model_name, N=512, model_type='bacon', hidden_layers=8,
                 hidden_size=256, output_layers=[1, 2, 4, 8],
                 return_sdf=False, adaptive=True):

    # the network has 4 output levels of detail
    num_outputs = len(output_layers)
    max_frequency = 3*(32,)

    # load model
    with utils.HiddenPrint():
        model = modules.MultiscaleBACON(3, hidden_size, 1,
                                        hidden_layers=hidden_layers,
                                        bias=True,
                                        frequency=max_frequency,
                                        quantization_interval=np.pi,
                                        is_sdf=True,
                                        output_layers=output_layers,
                                        reuse_filters=True)

    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    model.cuda()

    if not adaptive:
        # extracts separate meshes for each scale
        generate_mesh(model, N, return_sdf, num_outputs, model_name)

    else:
        # extracts single-scale output
        generate_mesh_adaptive(model, model_name)


def generate_mesh(model, N, return_sdf=False, num_outputs=4, model_name='model'):

    # write output
    x = torch.linspace(-0.5, 0.5, N)
    if return_sdf:
        x = torch.arange(-N//2, N//2) / N
        x = x.float()
    x, y, z = torch.meshgrid(x, x, x)
    render_coords = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=-1).cuda()
    sdf_values = [np.zeros((N**3, 1)) for i in range(num_outputs)]

    # render in a batched fashion to save memory
    bsize = int(128**2)
    for i in tqdm(range(int(N**3 / bsize))):
        coords = render_coords[i*bsize:(i+1)*bsize, :]
        out = model({'coords': coords})['model_out']

        if not isinstance(out, list):
            out = [out, ]

        for idx, sdf in enumerate(out):
            sdf_values[idx][i*bsize:(i+1)*bsize] = sdf.detach().cpu().numpy()

    if return_sdf:
        return [sdf.reshape(N, N, N) for sdf in sdf_values]

    for idx, sdf in enumerate(sdf_values):
        sdf = sdf.reshape(N, N, N)
        vertices, triangles = mcubes.marching_cubes(-sdf, 0)
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        mesh.vertices = (mesh.vertices / N - 0.5) + 0.5/N

        os.makedirs('./outputs/meshes',  exist_ok=True)
        mesh.export(f"./outputs/meshes/{model_name}_{idx+1}.obj")


def prepare_multi_scale(res, num_scales):
    def coord2ind(xyz_coord, res):
        # xyz_coord: * x 3
        x, y, z = torch.split(xyz_coord, 1, dim=-1)
        flat_ind = x * res**2 + y * res + z
        return flat_ind.squeeze(-1)  # *

    shifts = torch.from_numpy(np.stack(np.mgrid[:2, :2, :2], axis=-1)).view(-1, 3)

    def subdiv_index(xyz_prev, next_res):  # should output (N^3)*8
        xyz_next = xyz_prev.unsqueeze(1) * 2 + shifts  # (N^3)x8x3
        flat_ind_next = coord2ind(xyz_next, next_res)  # (N^3)*8
        return flat_ind_next

    lowest_res = res / 2**(num_scales-1)
    subdiv_hash_list = []

    for i in range(num_scales-1):
        curr_res = int(lowest_res*2**i)
        xyz_ind = torch.from_numpy(np.stack(np.mgrid[:curr_res, :curr_res, :curr_res], axis=-1)).view(-1, 3)  # (N^3)x3
        subdiv_hash = subdiv_index(xyz_ind, curr_res * 2)
        subdiv_hash_list.append(subdiv_hash.cuda().long())
    return subdiv_hash_list


# multi-scale marching cubes
def compute_one_scale(model, layer_ind, render_coords, sdf_values, hash_ind):
    assert(len(render_coords) == len(hash_ind))
    bsize = int(128 ** 2)
    for i in range(int(len(render_coords) / bsize)+1):
        coords = render_coords[i * bsize:(i + 1) * bsize, :]
        out = model({'coords': coords}, specified_layers=output_layers[layer_ind])['model_out']
        sdf_values[hash_ind[i * bsize:(i + 1) * bsize]] = out[0]


def compute_one_scale_adaptive(model, layer_ind, render_coords, sdf_values, hash_ind, threshold=0.003):
    assert(len(render_coords) == len(hash_ind))
    bsize = int(128 ** 2)
    for i in range(int(len(render_coords) / bsize)+1):
        coords = render_coords[i * bsize:(i + 1) * bsize, :]
        out = model({'coords': coords}, specified_layers=2, get_feature=True)['model_out']
        sdf = out[0][0]
        if output_layers[layer_ind] > 2:
            feature = out[0][1]
            near_surf = (sdf.abs() < threshold).squeeze()
            coords_surf = coords[near_surf]
            feature_surf = feature[near_surf]
            out = model({'coords': coords_surf}, specified_layers=output_layers[layer_ind],
                        continue_layer=2, continue_feature=feature_surf)['model_out']
            sdf_near = out[0]
            sdf[near_surf] = sdf_near

        sdf_values[hash_ind[i * bsize:(i + 1) * bsize]] = sdf


def generate_mesh_adaptive(model, model_name):
    with torch.no_grad():
        lowest_res = N / 2 ** (len(output_layers) - 1)
        compute_one_scale(model, 0, coords_list[0], sdf_out_list[0], subdiv_hashes[0])

        for i in range(1, num_outputs):
            curr_res = int(lowest_res*2**(i-1))
            pixel_len = 1 / curr_res
            threshold = (math.sqrt(2)*pixel_len*0.5)*2
            sdf_prev = sdf_out_list[i-1]
            sdf_curr = sdf_out_list[i]
            hash_curr = subdiv_hashes[i]
            coords_curr = coords_list[i]
            near_surf_prev = (sdf_prev.abs() <= threshold).squeeze(-1)

            # empty space
            sdf_curr[hash_curr[~near_surf_prev]] = sdf_prev[~near_surf_prev].unsqueeze(-1)

            # non-empty space
            non_empty_ind = hash_curr[near_surf_prev].flatten()

            if i == num_outputs-1:
                compute_one_scale_adaptive(model, i, coords_curr[non_empty_ind], sdf_curr,
                                           non_empty_ind, threshold=pixel_len*0.5*2.)
            else:
                compute_one_scale(model, i, coords_curr[non_empty_ind], sdf_curr, non_empty_ind)

        # run marching cubes
        sdf = sdf_curr.reshape(N, N, N).detach().cpu().numpy()
        vertices, triangles = mcubes.marching_cubes(-sdf, 0)
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        mesh.vertices = (mesh.vertices / N - 0.5) + 0.5/N

        os.makedirs('./outputs/meshes',  exist_ok=True)
        mesh.export(f"./outputs/meshes/{model_name}.obj")


def export_meshes(adaptive=True):
    bacon_ckpts = ['../trained_models/dragon.pth',
                   '../trained_models/armadillo.pth',
                   '../trained_models/lucy.pth',
                   '../trained_models/thai.pth']

    bacon_names = ['bacon_dragon',
                   'bacon_armadillo',
                   'bacon_lucy',
                   'bacon_thai']

    print('Exporting BACON')
    for ckpt, name in tqdm(zip(bacon_ckpts, bacon_names), total=len(bacon_ckpts)):
        export_model(ckpt, name, model_type='bacon', output_layers=output_layers, adaptive=adaptive)


def init_multiscale_mc():
    subdiv_hashes = prepare_multi_scale(N, len(output_layers))  # (N^3)*8
    subdiv_hashes = [torch.arange((N // 8) ** 3).cuda().long(), ] + subdiv_hashes
    lowest_res = N // 2**(len(output_layers)-1)
    coords_list = [dataio.get_mgrid(lowest_res*(2**i), dim=3).cuda() for i in range(len(output_layers))]  # (N^3)*3
    sdf_out_list = [torch.zeros(((lowest_res*(2**i))**3), 1).cuda() for i in range(len(output_layers))]  # (N^3)

    return subdiv_hashes, lowest_res, coords_list, sdf_out_list


if __name__ == '__main__':
    global N, output_layers, subdiv_hashes, lowest_res, coords_list, sdf_out_list, num_outputs
    N = 512
    output_layers = [2, 4, 6, 8]
    num_outputs = len(output_layers)

    subdiv_hashes, lowest_res, coords_list, sdf_out_list = init_multiscale_mc()

    # export meshes, use adaptive SDF evaluation or not
    # setting adaptive=False will output meshes at all resolutions
    # while adaptive=True while extract only a high-resolution mesh
    export_meshes(adaptive=True)
