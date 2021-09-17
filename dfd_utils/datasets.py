from torchvision import transforms, utils, datasets
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# TODO: Test this function
def get_transforms(img_height, img_width, size=None, p=0.5, crop=True, color_jitter=True, flip=True):
    if not size:
        size = img_height
    aspect_ratio = img_width / img_height
    if crop:
        crop = [transforms.transforms.RandomResizedCrop(size=(size, size), scale=(0.08, 1),
                                 ratio=((3. / 4) * aspect_ratio, (4. / 3) * aspect_ratio))]

    else:
        crop = [transforms.Resize(int(size * 1.5)),  # Added this since centercrop was cutting into the face
                transforms.CenterCrop(size)]
    if color_jitter:
        color_jitter = [transforms.RandomApply([transforms.ColorJitter(0.8 * p,
                                                                    0.8 * p,
                                                                    0.8 * p,
                                                                    0.2 * p)], p=0.8),
                                            transforms.RandomGrayscale(p=0.2)]
    else:
        color_jitter = []
    if flip:
        flip = [transforms.RandomHorizontalFlip(p=p)]
    else:
        flip = []
    to_tensor = [transforms.ToTensor()]
    transform_list = transforms.Compose(crop+color_jitter+flip+to_tensor)
    return transform_list

 # TODO: Add the FF dataset class
## SHIR: I think we don't need the seperate augment image function, see: https://github.com/leftthomas/SimCLR/blob/master/utils.py
class CIFAR10Pairs(datasets.CIFAR10):
  """
  CIFAR10 Dataset.
  returns 2 augmented pictures
  """
  def __getitem__(self, index):
    img, target = self.data[index], self.targets[index]
    img = Image.fromarray(img)
    if self.transform:
        pos_1 = self.transform(img)
        pos_2 = self.transform(img)
    return pos_1, pos_2, target


### TODO: add the transforms for the FF dataset.
TRANSFORMS = {'cifar_train_contrastive': get_transforms(img_height=32,img_width=32),
              'cifar_train_linear': get_transforms(img_height=32,img_width=32,crop=False, color_jitter=False),
              'cifar_test_linear': get_transforms(img_height=32,img_width=32,crop=False, 
                                                  color_jitter=False, flip=False)}

# Test change
def get_cifar_loaders(dataset_path, batch_size, contrastive):
    download = not os.path.exists(dataset_path)
    if contrastive:
        train_transforms = TRANSFORMS['cifar_train_contrastive']
    else:
        train_transforms = TRANSFORMS['cifar_train_linear']
    train_dataset = CIFAR10Pairs(root=dataset_path, train=True, download=download,
                                 transform=train_transforms)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = CIFAR10Pairs(root=dataset_path, train=False,
                                download=download, transform=TRANSFORMS['cifar_test_linear'])
    test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)
    cifar10_classes = ['airplane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return train_loader, test_loader, cifar10_classes


