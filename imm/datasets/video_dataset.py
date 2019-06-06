import os.path as osp
from os import listdir
import numpy as np
import tensorflow as tf

from imm.datasets.impair_dataset import ImagePairDataset

def load_dataset(data_root, subset):
    image_dir = osp.join(data_root, 'img')

    image_files = []
    future_image_files = []
    images_set = []

    # image, future image, train/val/test
    with open(osp.join(data_root, 'subsets', 'data_subsets.txt'), 'r') as f:
        lines = f.readlines()[1:]
        for i, line in enumerate(lines):
            data = line.split()
            image_files.append(data[0])
            future_image_files.append(data[1])
            images_set.append(int(data[2]))

    assert image_files[0] =='000000.png'

    if subset == 'train':
        label = 0
    elif subset == 'val':
        label = 1
    elif subset == 'test':
        label = 2
    else:
        raise ValueError(
            'subset = %s for video dataset not recognized.' % subset)

    images_set = np.array(images_set)
    image_files = np.array(image_files)
    future_image_files = np.array(future_image_files)
    images = image_files[images_set == label]
    future_images = future_image_files[images_set == label]

    return image_dir, images, future_images


class VideoDataset(ImagePairDataset):

    def __init__(self, data_dir, subset, max_samples=None, order_stream=False, image_size=[128,128], name='VideoDataset'):

        super(VideoDataset, self).__init__(
            data_dir, subset, image_size=image_size, jittering=None, name=name)

        self._image_dir, self._images, self._future_images = load_dataset(
            self._data_dir, self._subset)

        self._max_samples = max_samples
        self._order_stream = order_stream


    def _get_sample_dtype(self):
        d =  {'image': tf.string, 'future_image': tf.string}
        return d


    def _get_sample_shape(self):
        d = {'image': None,
             'future_image': None}
        return d


    def _get_image(self, idx):
        image = osp.join(self._image_dir, self._images[idx])
        future_image = osp.join(self._image_dir, self._future_images[idx])

        inputs = {'image': image, 'future_image': future_image}
        return inputs


    def _get_random_image(self):
        idx = np.random.randint(len(self._images))
        return self._get_image(idx)


    def _get_ordered_stream(self):
        for i in range(len(self._images)):
            yield self._get_image(i)


    def sample_image_pair(self):
        f_sample = self._get_random_image
        if self._order_stream:
            g = self._get_ordered_stream()
            f_sample = lambda: next(g)
        max_samples = float('inf')
        if self._max_samples is not None:
            max_samples = self._max_samples
        i_samp = 0
        while i_samp < max_samples:
            yield f_sample()
            if self._max_samples is not None:
                i_samp += 1


    def _get_smooth_step(self, n, b):
        x = tf.linspace(tf.cast(-1, tf.float32), 1, n)
        y = 0.5 + 0.5 * tf.tanh(x / b)
        return y


    def _get_smooth_mask(self, h, w, margin, step):
        b = 0.4
        step_up = self._get_smooth_step(step, b)
        step_down = self._get_smooth_step(step, -b)
        def create_strip(size):
            return tf.concat(
                [tf.zeros(margin, dtype=tf.float32),
                 step_up,
                 tf.ones(size - 2 * margin - 2 * step, dtype=tf.float32),
                 step_down,
                 tf.zeros(margin, dtype=tf.float32)], axis=0)
        mask_x = create_strip(w)
        mask_y = create_strip(h)
        mask2d = mask_y[:, None] * mask_x[None]
        return mask2d


    def _proc_im_pair(self, inputs):
        with tf.name_scope('proc_im_pair'):
            height, width = self._image_size[:2]

            # read in the images:
            image = self._read_image_tensor_or_string(inputs['image'])

            assert self._image_size[0] == self._image_size[1]
            final_size = self._image_size[0]

            image = tf.image.resize_images(
                image, [final_size, final_size], tf.image.ResizeMethod.BILINEAR,
                align_corners=True)

            mask = self._get_smooth_mask(height, width, 10, 20)[:, :, None]

            future_image = image

            inputs = {k: inputs[k] for k in self._get_sample_dtype().keys()}
            inputs.update({'image': image, 'future_image': future_image,
                           'mask': mask})
        return inputs


    def num_samples(self):
        raise NotImplementedError()
