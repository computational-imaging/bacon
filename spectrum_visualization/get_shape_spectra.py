import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules
import torch
import numpy as np
from tqdm import tqdm
from pykdtree.kdtree import KDTree
import mrcfile

device = torch.device('mps')


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def export_model(ckpt_path, model_name, N=512, model_type='bacon', hidden_layers=8,
                 hidden_size=256, output_layers=[1, 2, 4, 8], w0=30, pe=8,
                 filter_mesh=False, scaling=None, return_sdf=False):

    with HiddenPrints():
        # the network has 4 output levels of detail
        num_outputs = len(output_layers)
        max_frequency = 3*(32,)

        # load model
        if len(output_layers) > 1:
            model = modules.MultiscaleBACON(3, hidden_size, 1,
                                            hidden_layers=hidden_layers,
                                            bias=True,
                                            frequency=max_frequency,
                                            quantization_interval=np.pi,
                                            is_sdf=True,
                                            output_layers=output_layers,
                                            reuse_filters=True)

    print(model)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model = model.to(device)

    # write output
    x = torch.linspace(-0.5, 0.5, N)
    if return_sdf:
        x = torch.arange(-N//2, N//2) / N
        x = x.float()
    x, y, z = torch.meshgrid(x, x, x)
    render_coords = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=-1).to(device)
    sdf_values = [np.zeros((N**3, 1)) for i in range(num_outputs)]

    # render in a batched fashion to save memory
    bsize = int(128**2)
    for i in tqdm(range(int(N**3 / bsize))):
        coords = render_coords[i*bsize:(i+1)*bsize, :]
        out = model({'coords': coords})['model_out']

        if not isinstance(out, list):
            out = [out,]

        for idx, sdf in enumerate(out):
            sdf_values[idx][i*bsize:(i+1)*bsize] = sdf.detach().cpu().numpy()

    return [sdf.reshape(N, N, N) for sdf in sdf_values]


def normalize(coords, scaling=0.9):
    coords = np.array(coords).copy()
    cmean = np.mean(coords, axis=0, keepdims=True)
    coords -= cmean
    coord_max = np.amax(coords)
    coord_min = np.amin(coords)
    coords = (coords - coord_min) / (coord_max - coord_min)
    coords -= 0.5
    coords *= scaling

    scale = scaling / (coord_max - coord_min)
    offset = -scaling * (cmean + coord_min) / (coord_max - coord_min) - 0.5*scaling
    return coords, scale, offset


def get_ref_spectrum(xyz_file, N):
    pointcloud = np.genfromtxt(xyz_file)
    v = pointcloud[:, :3]
    n = pointcloud[:, 3:]

    n = n / (np.linalg.norm(n, axis=-1)[:, None])
    v, _, _ = normalize(v)
    print('loaded pc')

    # put pointcloud points into KDTree
    kd_tree = KDTree(v)
    print('made kd tree')

    # get sdf on grid and show
    x = (np.arange(-N//2, N//2) / N).astype(np.float32)
    coords = np.stack([arr.flatten() for arr in np.meshgrid(x, x, x)], axis=-1)

    sdf, idx = kd_tree.query(coords, k=3)

    # get average normal of hit point
    avg_normal = np.mean(n[idx], axis=1)
    sdf = np.sum((coords - v[idx][:, 0]) * avg_normal, axis=-1)
    sdf = sdf.reshape(N, N, N)
    return [sdf, ]


def extract_spectrum():

    scenes = ['armadillo']
    Ns = [384, 384, 384, 512, 512]
    methods = ['bacon', 'ref']

    for method in methods:
        for scene, N in zip(scenes, Ns):
            if method == 'ref':
                sdfs = get_ref_spectrum(f'ref_{scene}.xyz', N)
            else:

                ckpt = 'model_final.pth'

                sdfs = export_model(ckpt, scene, model_type=method, output_layers=[2, 4, 6, 8],
                                    return_sdf=True, N=N, pe=8, w0=30)

            sdfs_ft = [np.abs(np.fft.fftshift(np.fft.fftn(sdf))) for sdf in sdfs]
            sdfs_ft = [sdf_ft / np.max(sdf_ft)*1000 for sdf_ft in sdfs_ft]
            sdfs_ft = [np.clip(sdf_ft, 0, 1)**(1/3) for sdf_ft in sdfs_ft]

            for idx, sdf_ft in enumerate(sdfs_ft):
                with mrcfile.new_mmap(f'/tmp/sdf_ft_{idx}.mrc', overwrite=True, shape=(N, N, N), mrc_mode=2) as mrc:
                    mrc.data[:] = sdf_ft

            # render with chimera
            with open('/tmp/render.cxc', 'w') as f:
                for i in range(len(sdfs_ft)):
                    f.write(f'open /tmp/sdf_ft_{i}.mrc\n')
                    f.write('volume #1 style solid level 0,0 level 1,1\n')
                    f.write('volume #1 maximumIntensityProjection true\n')
                    f.write('volume #!1 showOutlineBox true\n')
                    f.write('volume #1 outlineBoxRgb slate gray\n')
                    f.write('volume #1 step 1\n')
                    f.write('view matrix camera 0.91721,0.10246,-0.38499,-533.37,-0.010261,0.97212,0.23427,631.78,0.39826,-0.21092,0.89269,1870.6\n')
                    f.write('view\n')
                    f.write(f'save /tmp/{method}_{scene}_{i+1}.png width 512 height 512 transparentBackground false\n')
                    f.write('close #1\n')
                f.write('exit\n')
            os.system('chimerax /tmp/render.cxc')

            for idx in range(len(sdfs_ft)):
                fname = f'{method}_{scene}_{idx+1}.png'
                os.system(f'./magicwand 1,1 -t 9 -r outside -m overlay -o 0 /tmp/{fname} {fname}')
                os.system(f'convert {fname} -trim +repage {fname}')
                os.remove(f'/tmp/{fname}')

    # clean up
    for idx in range(len(sdfs_ft)):
        os.remove(f'/tmp/sdf_ft_{idx}.mrc')
    os.remove('/tmp/render.cxc')


if __name__ == '__main__':
    extract_spectrum()
