# adapted from Jon Barron's mipnerf conversion script
# https://github.com/google/mipnerf/blob/main/scripts/convert_blender_data.py

import json
import os
from os import path

from absl import app
from absl import flags
import numpy as np
from PIL import Image

import skimage.transform

FLAGS = flags.FLAGS

flags.DEFINE_string('blenderdir', None, 'Base directory for all Blender data.')
flags.DEFINE_integer('n_down', 4, 'How many levels of downscaling to use.')


def load_renderings(data_dir, split):
    """Load images and metadata from disk."""
    f = 'transforms_{}.json'.format(split)
    with open(path.join(data_dir, f), 'r') as fp:
        meta = json.load(fp)
    images = []
    cams = []
    print('Loading imgs')
    for frame in meta['frames']:
        fname = os.path.join(data_dir, frame['file_path'] + '.png')
        with open(fname, 'rb') as imgin:
            image = np.array(Image.open(imgin), dtype=np.float32) / 255.
        cams.append(frame['transform_matrix'])
        images.append(image)
    ret = {}
    ret['images'] = np.stack(images, axis=0)
    print('Loaded all images, shape is', ret['images'].shape)
    ret['camtoworlds'] = np.stack(cams, axis=0)
    w = ret['images'].shape[2]
    camera_angle_x = float(meta['camera_angle_x'])
    ret['focal'] = .5 * w / np.tan(.5 * camera_angle_x)
    return ret


def down2(img):
    sh = img.shape
    return np.mean(np.reshape(img, [sh[0] // 2, 2, sh[1] // 2, 2, -1]), (1, 3))


def convert_to_nerfdata(basedir, n_down):
    """Convert Blender data to multiscale."""
    splits = ['train', 'val', 'test']
    # Foreach split in the dataset
    for split in splits:
        print('Split', split)
        # Load everything
        data = load_renderings(basedir, split)

        # Save out all the images
        imgdir = '{}_multiscale'.format(split)
        os.makedirs(os.path.join(basedir, imgdir), exist_ok=True)
        print('Saving images')
        for i, img in enumerate(data['images']):
            for j in range(n_down):
                fname = '{}/r_{:d}_d{}.png'.format(imgdir, i, j)
                fname = os.path.join(basedir, fname)
                with open(fname, 'wb') as imgout:
                    img = skimage.transform.resize(img, 2*(512//2**j,))
                    img8 = Image.fromarray(np.uint8(img * 255))
                    img8.save(imgout)
                # img = down2(img)


def main(unused_argv):

    blenderdir = FLAGS.blenderdir
    n_down = FLAGS.n_down

    dirs = [os.path.join(blenderdir, f) for f in os.listdir(blenderdir)]
    dirs = [d for d in dirs if os.path.isdir(d)]
    print(dirs)
    for basedir in dirs:
        print()
        convert_to_nerfdata(basedir, n_down)


if __name__ == '__main__':
    app.run(main)
