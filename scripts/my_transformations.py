import torch
import numpy as np
import random
from torchvision.transforms import ColorJitter, RandomEqualize, RandomAutocontrast, GaussianBlur, RandomAdjustSharpness

# LIST_OF_TRANSFORMS = [ColorJitter, RandomEqualize, RandomAutocontrast, GaussianBlur, RandomAdjustSharpness]
LIST_OF_TRANSFORMS = [ColorJitter, RandomEqualize, RandomAutocontrast]

def extra_transforms(image):
    for transform in LIST_OF_TRANSFORMS:
        if random.random() < 0.25:
            image = transform(image)

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample
        

        h, w = image.shape[-2:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        
        print(top,left)

        image = image[:,top: top + new_h,
                      left: left + new_w]
        
        mask = mask[top: top + new_h,
                      left: left + new_w]


        return image, mask


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}