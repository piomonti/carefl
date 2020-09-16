import numpy as np
import os
import torch
import torchvision
from skimage import io, transform
from torch.utils.data import Dataset

from models import CAReFl


class Rescale:
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, images):
        h, w = images[0].shape
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        return [transform.resize(image, (new_h, new_w)).squeeze() for image in images]


class Crop:
    """Center crop the image in a sample to a given size.

    Args:
        output_size int: Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size, random=False):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.random = random

    def __call__(self, images):
        h, w = images[0].shape
        if isinstance(self.output_size, int):
            new_h = new_w = self.output_size
        else:
            new_h, new_w = self.output_size
        if self.random:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
        else:
            top, left = h // 2 - new_h // 2, w // 2 - new_w // 2
        return [image[top: top + new_h, left: left + new_w] for image in images]


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, images):
        return [torch.from_numpy(image) for image in images]


class Flatten:
    """FLatten images and combine into one long vector"""

    def __call__(self, images):
        if isinstance(images[0], torch.Tensor):
            return torch.cat([image.reshape(-1) for image in images])
        else:
            return np.concatenate([image.reshape(-1) for image in images])


class SingleVideoDataset(Dataset):
    """A dataset to load a video for arrow of time detection"""

    def __init__(self, video_idx=None, video_name=None, transform=None):
        """
        Args:
            video_idx (int): Index of the video to load
            video_name (string): Alternatively, directly give the name of the video
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = '/nfs/data/ilyesk/arrow/'
        video_path = os.path.join(self.root, 'ArrowDataAll')
        names = os.listdir(self.video_path)
        if video_idx is None and video_name is None:
            raise ValueError('Please specify a video index or name')
        if video_idx is not None and video_name is not None:
            raise ValueError('Please only specify a video index OR name')
        if video_name is not None:
            try:
                idx = np.argwhere([video_name in x for x in self.names])[0, 0]
            except IndexError:
                raise ValueError('Video not in dataset')
        else:
            idx = video_idx
        self.video_idx = idx
        self.video_name = self.names[idx]
        self.path = os.path.join(self.video_path, self.video_name)
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.path))

    def __getitem__(self, idx):

        image_path = os.path.join(self.path, 'im{:08}.jpeg'.format(idx + 1))
        im = io.imread(image_path, as_gray=True).astype(np.float32)
        if self.transform:
            im = self.transform(im)
        return im


class ArrowDataset(Dataset):
    """A dataset to load a video and return a couple of successive frames"""

    def __init__(self, video_idx=None, video_name=None, transform=None, lag=1, split=1.):
        """
        Args:
            video_idx (int): Index of the video to load
            video_name (string): Alternatively, directly give the name of the video
            transform (callable, optional): Optional transform to be applied on a sample.
            lag: lag between the returned frames
        """
        self.root = '/nfs/data/ilyesk/arrow/'
        self.video_path = os.path.join(self.root, 'ArrowDataAll')
        self.names = os.listdir(self.video_path)
        if video_idx is None and video_name is None:
            raise ValueError('Please specify a video index or name')
        if video_idx is not None and video_name is not None:
            raise ValueError('Please only specify a video index OR name')
        if video_name is not None:
            try:
                idx = np.argwhere([video_name in x for x in self.names])[0, 0]
            except IndexError:
                raise ValueError('Video not in dataset')
        else:
            idx = video_idx
        self.video_idx = idx
        self.video_name = self.names[idx]
        self.path = os.path.join(self.video_path, self.video_name)
        self.lag = lag
        self.transform = transform
        self.split = split

    def __len__(self):
        length = len(os.listdir(self.path)) - self.lag
        return int(self.split * length)

    def __getitem__(self, idx):
        im1_path = os.path.join(self.path, 'im{:08}.jpeg'.format(idx + 1))
        im2_path = os.path.join(self.path, 'im{:08}.jpeg'.format(idx + 1 + self.lag))
        im1 = io.imread(im1_path, as_gray=True).astype(np.float32)
        im2 = io.imread(im2_path, as_gray=True).astype(np.float32)
        transformed_image = self.transform((im1, im2)) if self.transform else (im1, im2)
        # return torch.cat((im1.view(-1), im2.view(-1)))
        return transformed_image


def video_runner(args, config):
    # read two consecutive frames of a video, and transform them into one long vector
    tran = torchvision.transforms.Compose([Rescale(128),
                                           Crop(64, random=True),
                                           ToTensor(),
                                           Flatten()])
    # each of these datasets returns vectors of the form [X, Y] where X is a flattened frame in a video that
    # precedes the flattened frame Y. so the causal direction is always X -> Y
    dset = ArrowDataset(video_idx=0, transform=tran)
    test_dset = ArrowDataset(video_idx=0, split=.5, transform=tran)
    # load a CAReFl model
    model = CAReFl(config)
    # predict_proba takes one argument, pack dsets into a tuple
    p, dir = model.predict_proba((dset, test_dset))
    print(dir == 'x->y')
