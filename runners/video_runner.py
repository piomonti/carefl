import numpy as np
import os
import pickle
import torch
from skimage import io
from skimage import transform as sktransform
from sklearn.decomposition import PCA
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
        return [sktransform.resize(image, (new_h, new_w)).squeeze() for image in images]


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
        self.root = '/nfs/gatsbystor/ilyesk/arrow/'
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
        self.root = '/nfs/gatsbystor/ilyesk/arrow/'
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


class VideoFeatures:
    def __init__(self, config, train=True, transform=None, pca=False, n_components=1):
        """
        Args:
            config: config file containing video_idx, lag, and split
            transform: Optional transform to be applied on a sample.
            pca: whether to apply pca to the features
            n_components: number of components to keep in the pca
        """
        self.root = 'data/video/'
        self.config = config
        self.video_idx = config.data.video_idx
        self.lag = config.data.lag
        self.split = config.training.split
        self.train = train

        raw_data = torch.load(os.path.join(self.root, 'video_{}_{}_{}.pt'.format(config.data.video_idx,
                                                                                 config.data.image_size,
                                                                                 config.data.crop_size)))
        raw_data = raw_data.detach()
        if pca:
            p = PCA(n_components=n_components)
            raw_data = torch.from_numpy(p.fit_transform(raw_data.numpy()))

        if isinstance(self.lag, int):
            self.lag = [self.lag]
        data = []
        for lag in self.lag:
            data.append(torch.cat([raw_data[:-lag], raw_data[lag:]], dim=1))
        data = torch.cat(data, dim=0)

        self.raw_data = raw_data

        if self.split < 1:
            self.data = data[:int(self.split * len(data))] if train else data[int(self.split * len(data)):]
        else:
            self.data = data

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


def res_save_name(args, config):
    pca = 'f' if not config.data.pca else 't{}'.format(config.data.n_components)
    return 'vid_{}_{}_{}_{}__{}_{}_{}_{}_{}.p'.format(config.data.image_size,
                                                      config.data.crop_size,
                                                      pca,
                                                      config.data.lag,
                                                      config.flow.architecture.lower(),
                                                      config.flow.net_class.lower(),
                                                      config.flow.nl,
                                                      config.flow.nh,
                                                      args.seed)


def video_runner(args, config):
    # each of these datasets returns vectors of features of the form [X, Y] where X and Y are features of frames of a
    # video computed using GoogLeNet, such that X precedes Y in the video
    train_dset = VideoFeatures(config, train=True, pca=config.data.pca, n_components=config.data.n_components)
    test_dset = VideoFeatures(config, train=False, pca=config.data.pca, n_components=config.data.n_components)
    # load a CAReFl model
    model = CAReFl(config)
    # predict_proba takes one argument, pack dsets into a tuple
    p, direction = model.predict_proba((train_dset, test_dset))
    true_dir = 'x->y' if config.data.video_idx < 155 else 'y->x'
    print(direction == true_dir)
    result = {'p': p, 'dir': direction, 'c': direction == true_dir}
    path = os.path.join(args.output, config.data.video_idx)
    os.makedirs(path, exist_ok=True)
    pickle.dump(result, open(os.path.join(path, res_save_name(args, config)), 'wb'))


def plot_video(args, config):
    cs = []
    ps = []
    for seed in range(20):
        args.seed = seed
        res = pickle.load(open(os.path.join(args.output, config.data.video_idx, res_save_name(args, config)), 'rb'))
        cs.append(res['c'])
        ps.append(res['p'])
    print("Average correct:", np.mean(cs))
    print("sequence of p's:", ps)
