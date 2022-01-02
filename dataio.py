import torch
import numpy as np
import math
from PIL import Image
import skimage
from torchvision.transforms import Compose, ToTensor, Resize, Lambda
import skimage.transform
import json
import os
import re
from tqdm import tqdm
from torch.utils.data import Dataset
from pykdtree.kdtree import KDTree
import errno
import urllib.request


def get_mgrid(sidelen, dim=2, centered=True, include_end=False):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if include_end:
        denom = [s-1 for s in sidelen]
    else:
        denom = sidelen

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / denom[0]
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / denom[1]
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / denom[0]
        pixel_coords[..., 1] = pixel_coords[..., 1] / denom[1]
        pixel_coords[..., 2] = pixel_coords[..., 2] / denom[2]
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    if centered:
        pixel_coords -= 0.5

    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)


class Func1DWrapper(torch.utils.data.Dataset):
    def __init__(self, range, fn, grad_fn=None,
                 sampling_density=100, train_every=10):

        coords = self.get_samples(range, sampling_density)
        self.fn_vals = fn(coords)
        self.train_idx = torch.arange(0, coords.shape[0], train_every).float()

        self.grid = coords
        self.grid.requires_grad_(True)
        self.range = range

    def get_samples(self, range, sampling_density):
        num = int(range[1] - range[0])*sampling_density
        coords = np.linspace(start=range[0], stop=range[1], num=num)
        coords.astype(np.float32)
        coords = torch.Tensor(coords).view(-1, 1)
        return coords

    def get_num_samples(self):
        return self.grid.shape[0]

    def __len__(self):
        return 1

    def __getitem__(self, idx):

        return {'idx': self.train_idx, 'coords': self.grid}, \
               {'func': self.fn_vals, 'coords': self.grid}


def rect(coords, width=1):
    return torch.where(abs(coords) < width/2, 1.0/width, 0.0)


def gaussian(coords, sigma=1, center=0.5):
    return 1 / (sigma * math.sqrt(2*np.pi)) * torch.exp(-(coords-center)**2 / (2*sigma**2))


def sines1(coords):
    return 0.3 * torch.sin(2*np.pi*8*coords + np.pi/3) + 0.65 * torch.sin(2*np.pi*2*coords + np.pi)


def polynomial_1(coords):
    return .1*((coords+.2)*3)**5 - .2*((coords+.2)*3)**4 + .2*((coords+.2)*3)**3 - .4*((coords+.2)*3)**2 + .1*((coords+.2)*3)


def sinc(coords):
    coords[coords == 0] += 1
    return torch.div(torch.sin(20*coords), 20*coords)


def linear(coords):
    return 1.0 * coords


def xcosx(coords):
    return coords * torch.cos(coords)


class ImageWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, compute_diff='all', centered=True,
                 include_end=False, multiscale=False, stages=3):

        self.compute_diff = compute_diff
        self.centered = centered
        self.include_end = include_end
        self.transform = Compose([
            ToTensor(),
        ])

        self.dataset = dataset
        self.mgrid = get_mgrid(self.dataset.resolution, centered=centered, include_end=include_end)

        # sample pixel centers
        self.mgrid = self.mgrid + 1 / (2 * self.dataset.resolution[0])
        self.radii = 1 / self.dataset.resolution[0] * 2/np.sqrt(12)
        self.radii = [(self.radii * 2**i).astype(np.float32) for i in range(3)]
        self.radii.reverse()

        img = self.transform(self.dataset[0])
        _, self.rows, self.cols = img.shape

        self.img_chw = img
        self.img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)

        self.imgs = []
        self.multiscale = multiscale
        img = img.permute(1, 2, 0).numpy()
        for i in range(stages):
            tmp = skimage.transform.resize(img, [s//2**i for s in (self.rows, self.cols)])
            tmp = skimage.transform.resize(tmp, (self.rows, self.cols))
            self.imgs.append(torch.from_numpy(tmp).view(-1, self.dataset.img_channels))
        self.imgs.reverse()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        coords = self.mgrid
        img = self.img

        in_dict = {'coords': coords, 'radii': self.radii}
        gt_dict = {'img': img}

        if self.multiscale:
            gt_dict['img'] = self.imgs

        return in_dict, gt_dict


def save_img(img, filename):
    ''' given np array, convert to image and save '''
    img = Image.fromarray((255*img).astype(np.uint8))
    img.save(filename)


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))


class ImageFile(Dataset):
    def __init__(self, filename, grayscale=False, resolution=None,
                 root_path=None, crop_square=True, url=None):

        super().__init__()

        if not os.path.exists(filename):
            if url is None:
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), filename)
            else:
                print('Downloading image file...')
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                urllib.request.urlretrieve(url, filename)

        self.img = Image.open(filename)
        if grayscale:
            self.img = self.img.convert('L')
        else:
            self.img = self.img.convert('RGB')

        self.img_channels = len(self.img.mode)
        self.resolution = self.img.size

        if crop_square:  # preserve aspect ratio
            self.img = crop_max_square(self.img)

        if resolution is not None:
            self.resolution = resolution
            self.img = self.img.resize(resolution, Image.ANTIALIAS)

        self.img = np.array(self.img)
        self.img = self.img.astype(np.float32)/255.

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.img


def chunk_lists_from_batch_reduce_to_raysamples_fn(model_input, meta, gt, max_chunk_size):

    model_in_chunked = []
    for key in model_input:
        num_views, num_rays, num_samples_per_rays, num_dims = model_input[key].shape
        chunks = torch.split(model_input[key].view(-1, num_samples_per_rays, num_dims), max_chunk_size)
        model_in_chunked.append(chunks)

    list_chunked_model_input = \
        [{k: v for k, v in zip(model_input.keys(), curr_chunks)} for curr_chunks in zip(*model_in_chunked)]

    # meta_dict
    list_chunked_zs = torch.split(meta['zs'].view(-1, num_samples_per_rays, 1),
                                  max_chunk_size)
    list_chunked_meta = [{'zs': zs} for zs in list_chunked_zs]

    # gt_dict
    gt_chunked = []
    for key in gt:
        if isinstance(gt[key], list):
            # this handles lists of gt tensors (e.g., for multiscale)
            num_dims = gt[key][0].shape[-1]

            # this chunks the list elements so you have [num_tensors, num_chunks]
            chunks = [torch.split(x.view(-1, num_dims), max_chunk_size) for x in gt[key]]

            # this switches it to [num_chunks, num_tensors]
            chunks = [chunk for chunk in zip(*chunks)]
            gt_chunked.append(chunks)
        else:
            *_, num_dims = gt[key].shape
            chunks = torch.split(gt[key].view(-1, num_dims), max_chunk_size)
            gt_chunked.append(chunks)

    list_chunked_gt = \
        [{k: v for k, v in zip(gt.keys(), curr_chunks)} for curr_chunks in zip(*gt_chunked)]

    return list_chunked_model_input, list_chunked_meta, list_chunked_gt


class NerfBlenderDataset(torch.utils.data.Dataset):
    def __init__(self, basedir, mode='train',
                 splits=['train', 'val', 'test'],
                 select_idx=None,
                 testskip=1, resize_to=None, final_render=False,
                 d_rot=0, bounds=((-2, 2), (-2, 2), (0, 2)),
                 multiscale=False,
                 black_background=False,
                 override_scale=None):

        self.mode = mode
        self.basedir = basedir
        self.resize_to = resize_to
        self.final_render = final_render
        self.bounds = bounds
        self.multiscale = multiscale
        self.select_idx = select_idx
        self.d_rot = d_rot

        metas = {}
        for s in splits:
            with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
                metas[s] = json.load(fp)

        # Eventually transform the inputs
        transform_list = [ToTensor()]
        if resize_to is not None:
            transform_list.insert(0, Resize(resize_to,
                                            interpolation=Image.BILINEAR))

        def multiscale_resize(x):
            scale = 512 // x.size[0]
            return x.resize([r//scale for r in resize_to],
                            resample=Image.BILINEAR)

        if multiscale and override_scale is None:
            # this will scale the image down appropriately
            # (e.g., to 1/2, 1/4, 1/8 of desired resolution)
            # Then the next transform will scale it back up so we use the same rays
            # to supervise
            transform_list.insert(0, Lambda(lambda x: multiscale_resize(x)))
        if black_background:
            transform_list.append(Lambda(lambda x: x[:3] * x[[-1]]))
        else:
            transform_list.append(Lambda(lambda x: x[:3] * x[[-1]] + (1 - x[[-1]])))

        self.transforms = Compose(transform_list)

        # Gather images and poses
        self.all_imgs = {}
        self.all_poses = {}
        for s in splits:
            meta = metas[s]
            imgs, poses = self.load_images(s, meta, testskip)

            self.all_imgs.update({s: imgs})
            self.all_poses.update({s: poses})

        if self.final_render:
            self.poses = [torch.from_numpy(self.pose_spherical(angle, -30.0, 4.0)).float()
                          for angle in np.linspace(-180, 180, 40 + 1)[:-1]]

        if override_scale is not None:
            assert multiscale, 'only for multiscale'
            if override_scale > 3:
                override_scale = 3
            H, W = self.multiscale_imgs[0][override_scale].shape[:2]
            self.img_shape = self.multiscale_imgs[0][override_scale].shape
        else:
            H, W = imgs[0].shape[:2]
            self.img_shape = imgs[0].shape

        # projective camera
        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        self.camera_params = {'H': H, 'W': W,
                              'camera_angle_x': camera_angle_x,
                              'focal': focal,
                              'near': 2.0,
                              'far': 6.0}

    def load_images(self, s, meta, testskip):
        imgs = []
        poses = []

        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in tqdm(meta['frames'][::skip]):
            if self.select_idx is not None:
                if re.search('[0-9]+', frame['file_path']).group(0) != self.select_idx:
                    continue

            def load_image(fname):
                img = Image.open(fname)
                pose = torch.from_numpy(np.array(frame['transform_matrix'], dtype=np.float32))

                img_t = self.transforms(img)
                imgs.append(img_t.permute(1, 2, 0))
                poses.append(pose)

            if self.multiscale:
                for i in range(4):
                    fname = os.path.join(self.basedir, frame['file_path']).replace(s, s + '_multiscale') + f'_d{i}.png'
                    load_image(fname)

            else:
                fname = os.path.join(self.basedir, frame['file_path'] + '.png')
                load_image(fname)

        if self.multiscale:
            poses = poses[::4]
            self.multiscale_imgs = [imgs[i:i+4][::-1] for i in range(0, len(imgs), 4)]
            imgs = imgs[::4]

        return imgs, poses

    # adapted from https://github.com/krrish94/nerf-pytorch
    # derived from original NeRF repo (MIT License)
    def translate_by_t_along_z(self, t):
        tform = np.eye(4).astype(np.float32)
        tform[2][3] = t
        return tform

    def rotate_by_phi_along_x(self, phi):
        tform = np.eye(4).astype(np.float32)
        tform[1, 1] = tform[2, 2] = np.cos(phi)
        tform[1, 2] = -np.sin(phi)
        tform[2, 1] = -tform[1, 2]
        return tform

    def rotate_by_theta_along_y(self, theta):
        tform = np.eye(4).astype(np.float32)
        tform[0, 0] = tform[2, 2] = np.cos(theta)
        tform[0, 2] = -np.sin(theta)
        tform[2, 0] = -tform[0, 2]
        return tform

    def pose_spherical(self, theta, phi, radius):
        c2w = self.translate_by_t_along_z(radius)
        c2w = self.rotate_by_phi_along_x(phi / 180.0 * np.pi) @ c2w
        c2w = self.rotate_by_theta_along_y(theta / 180 * np.pi) @ c2w
        c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
        return c2w

    def set_mode(self, mode):
        self.mode = mode

    def get_img_shape(self):
        return self.img_shape

    def get_camera_params(self):
        return self.camera_params

    def __len__(self):
        if self.final_render:
            return len(self.poses)
        else:
            return len(self.all_imgs[self.mode])

    def __getitem__(self, item):
        # render out trajectory (no GT images)
        if self.final_render:
            return {'img': torch.zeros(4),  # we have to pass something...
                    'pose': self.poses[item]}

        # otherwise, return GT images and pose
        else:
            return {'img': self.all_imgs[self.mode][item],
                    'pose': self.all_poses[self.mode][item]}


class Implicit6DMultiviewDataWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, img_shape, camera_params,
                 samples_per_ray=128,
                 samples_per_view=32000,
                 num_workers=4,
                 multiscale=False,
                 supervise_hr=False,
                 scales=[1/8, 1/4, 1/2, 1]):

        self.dataset = dataset
        self.num_workers = num_workers
        self.multiscale = multiscale
        self.scales = scales
        self.supervise_hr = supervise_hr

        self.img_shape = img_shape
        self.camera_params = camera_params

        self.samples_per_view = samples_per_view
        self.default_samples_per_view = samples_per_view
        self.samples_per_ray = samples_per_ray

        self._generate_rays_normalized()
        self._precompute_rays()

        self.is_logging = False

        self.val_idx = 0

        self.num_rays = self.all_ray_orgs.view(-1, 3).shape[0]

        self.shuffle_rays()

        if multiscale:
            self.multiscale_imgs = dataset.multiscale_imgs

            # switch to size [num_scales, num_views, img_size[0], img_size[1], 3]
            self.multiscale_imgs = torch.stack([torch.stack(m, dim=0)
                                                for m in zip(*self.multiscale_imgs)], dim=0)

    def toggle_logging_sampling(self):
        if self.is_logging:
            self.samples_per_view = self.default_samples_per_view
            self.is_logging = False
        else:
            self.samples_per_view = self.img_shape[0] * self.img_shape[1]
            self.is_logging = True

    def _generate_rays_normalized(self):

        # projective camera
        rows = torch.arange(0, self.img_shape[0], dtype=torch.float32)
        cols = torch.arange(0, self.img_shape[1], dtype=torch.float32)
        g_rows, g_cols = torch.meshgrid(rows, cols)

        W = self.camera_params['W']
        H = self.camera_params['H']
        f = self.camera_params['focal']

        self.norm_rays = torch.stack([(g_cols-.5*W + 0.5)/f,
                                      -(g_rows-.5*H + 0.5)/f,
                                      -torch.ones_like(g_rows)],
                                     dim=2).view(-1, 3).permute(1, 0)

        self.num_rays_per_view = self.norm_rays.shape[1]

    def shuffle_rays(self):
        self.shuffle_idxs = torch.randperm(self.num_rays)

    def _precompute_rays(self):
        img_list = []
        pose_list = []
        ray_orgs_list = []
        ray_dirs_list = []

        print('Precomputing rays...')
        for img_pose in tqdm(self.dataset):
            img = img_pose['img']
            img_list.append(img)

            pose = img_pose['pose']
            pose_list.append(pose)

            ray_dirs = pose[:3, :3].matmul(self.norm_rays).permute(1, 0)
            ray_dirs_list.append(ray_dirs)

            ray_orgs = pose[:3, 3].repeat((self.num_rays_per_view, 1))
            ray_orgs_list.append(ray_orgs)

        self.all_imgs = torch.stack(img_list, dim=0)
        self.all_poses = torch.stack(pose_list, dim=0)
        self.all_ray_orgs = torch.stack(ray_orgs_list, dim=0)
        self.all_ray_dirs = torch.stack(ray_dirs_list, dim=0)

        self.hit = torch.zeros(self.all_ray_dirs.view(-1, 3).shape[0])

    def __len__(self):
        if self.is_logging:
            return self.all_imgs.shape[0]
        else:
            return self.num_rays // self.samples_per_view

    def get_val_rays(self):
        img = self.all_imgs[self.val_idx, ...]
        ray_dirs = self.all_ray_dirs[self.val_idx, ...]
        ray_orgs = self.all_ray_orgs[self.val_idx, ...]
        view_samples = img

        if self.multiscale:
            img = self.multiscale_imgs[:, self.val_idx, ...]
            if self.supervise_hr:
                img = [img[-1] for _ in img]
            view_samples = [im for im in img]

        self.val_idx += 1
        self.val_idx %= self.all_imgs.shape[0]

        return view_samples, ray_orgs, ray_dirs

    def get_rays(self, idx):
        idxs = self.shuffle_idxs[self.samples_per_view * idx:self.samples_per_view * (idx+1)]
        ray_dirs = self.all_ray_dirs.view(-1, 3)[idxs, ...]
        ray_orgs = self.all_ray_orgs.view(-1, 3)[idxs, ...]

        if self.multiscale:
            view_samples = [mimg.view(-1, 3)[idxs] for mimg in self.multiscale_imgs]

            if self.supervise_hr:
                view_samples = [view_samples[-1] for _ in view_samples]
        else:
            img = self.all_imgs.view(-1, 3)[idxs, ...]
            view_samples = img.reshape(-1, 3)

        self.hit[idxs] += 1

        return view_samples, ray_orgs, ray_dirs

    def __getitem__(self, idx):

        if self.is_logging:
            view_samples, ray_orgs, ray_dirs = self.get_val_rays()
        else:
            view_samples, ray_orgs, ray_dirs = self.get_rays(idx)

        # Transform coordinate systems
        camera_params = self.dataset.get_camera_params()

        ray_dirs = ray_dirs[:, None, :]
        ray_orgs = ray_orgs[:, None, :]

        t_vals = torch.linspace(0.0, 1.0, self.samples_per_ray)
        t_vals = camera_params['near'] * (1.0 - t_vals) + camera_params['far'] * t_vals
        t_vals = t_vals[None, :].repeat(self.samples_per_view, 1)

        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat((mids, t_vals[..., -1:]), dim=-1)
        lower = torch.cat((t_vals[..., :1], mids), dim=-1)

        # Stratified samples in those intervals.
        t_rand = torch.rand(t_vals.shape)
        t_vals = lower + (upper - lower) * t_rand

        ray_samples = ray_orgs + ray_dirs * t_vals[..., None]

        t_intervals = t_vals[..., 1:] - t_vals[..., :-1]
        t_intervals = torch.cat((t_intervals, 1e10*torch.ones_like(t_intervals[:, 0:1])), dim=-1)
        t_intervals = (t_intervals * ray_dirs.norm(p=2, dim=-1))[..., None]

        # Compute distance samples from orgs
        dist_samples_to_org = torch.sqrt(torch.sum((ray_samples-ray_orgs)**2, dim=-1, keepdim=True))

        # broadcast tensors
        view_dirs = ray_dirs / ray_dirs.norm(p=2, dim=-1, keepdim=True).repeat(1, self.samples_per_ray, 1)

        in_dict = {'ray_samples': ray_samples,
                   'ray_orientations': view_dirs,
                   'ray_origins': ray_orgs,
                   't_intervals': t_intervals,
                   't': t_vals[..., None],
                   'ray_directions': ray_dirs}
        meta_dict = {'zs': dist_samples_to_org}

        gt_dict = {'pixel_samples': view_samples}

        return in_dict, meta_dict, gt_dict


class MeshSDF(Dataset):
    ''' convert point cloud to SDF '''

    def __init__(self, pointcloud_path, num_samples=30**3,
                 coarse_scale=1e-1, fine_scale=1e-3):
        super().__init__()
        self.num_samples = num_samples
        self.pointcloud_path = pointcloud_path
        self.coarse_scale = coarse_scale
        self.fine_scale = fine_scale

        self.load_mesh(pointcloud_path)

    def __len__(self):
        return 10000  # arbitrary

    def load_mesh(self, pointcloud_path):
        pointcloud = np.genfromtxt(pointcloud_path)
        self.v = pointcloud[:, :3]
        self.n = pointcloud[:, 3:]

        n_norm = (np.linalg.norm(self.n, axis=-1)[:, None])
        n_norm[n_norm == 0] = 1.
        self.n = self.n / n_norm
        self.v = self.normalize(self.v)
        self.kd_tree = KDTree(self.v)
        print('loaded pc')

    def normalize(self, coords):
        coords -= np.mean(coords, axis=0, keepdims=True)
        coord_max = np.amax(coords)
        coord_min = np.amin(coords)
        coords = (coords - coord_min) / (coord_max - coord_min) * 0.9
        coords -= 0.45
        return coords

    def sample_surface(self):
        idx = np.random.randint(0, self.v.shape[0], self.num_samples)
        points = self.v[idx]
        points[::2] += np.random.laplace(scale=self.coarse_scale, size=(points.shape[0]//2, points.shape[-1]))
        points[1::2] += np.random.laplace(scale=self.fine_scale, size=(points.shape[0]//2, points.shape[-1]))

        # wrap around any points that are sampled out of bounds
        points[points > 0.5] -= 1
        points[points < -0.5] += 1

        # use KDTree to get distance to surface and estimate the normal
        sdf, idx = self.kd_tree.query(points, k=3)
        avg_normal = np.mean(self.n[idx], axis=1)
        sdf = np.sum((points - self.v[idx][:, 0]) * avg_normal, axis=-1)
        sdf = sdf[..., None]

        return points, sdf

    def __getitem__(self, idx):
        coords, sdf = self.sample_surface()

        return {'coords': torch.from_numpy(coords).float()}, \
               {'sdf': torch.from_numpy(sdf).float()}
