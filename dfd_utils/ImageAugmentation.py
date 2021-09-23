import torch
from torchvision import transforms
import numpy as np

def random_crop_resize_image(image, height, width):
    """
    image should be ndarry
    """
    aspect_ratio = width/height
    train_trans = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.RandomResizedCrop(size=(height, width), scale=(0.08, 1), 
                                                                ratio=((3./4)*aspect_ratio, (4./3)*aspect_ratio)),
    ]) 
    # SHIR: why isn't the scale 0.08-1.0 as in the tutorial?
    # I added the aspect ratio range as appears on the finetuning tutorial of simclr
    # LEV: When playing with the FF dataset, you can see that sometimes the faces are cutoff, im trying to find a 
    # configuration which will leave the entire face, 
    new_img = train_trans(image)
    return np.array(new_img)

def color_distort_image(image, p=0.5):
    """
    image should be ndarry
    """
    train_trans = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.RandomApply([
                                                            transforms.ColorJitter(0.8*p, 
                                                              0.8*p, 
                                                              0.8*p, 
                                                              0.2*p)], p = 0.8),
                                    transforms.RandomGrayscale(p=0.2)
    ])
    new_img = train_trans(image)
    return np.array(new_img)

def random_flip_image(image, p=0.5):
    '''
    image should be ndarry
    '''
    train_trans = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.RandomHorizontalFlip(p=p),
    ])
    new_img = train_trans(image)
    return np.array(new_img)


def augment_image(image, size, crop=True, flip=True, color_distort=True):
    if type(image) is not np.ndarray:
        image = np.array(image)
    #image = np.uint8(image) # For some reason ToPil image doesnt like u32
    #img = transforms.ToPILImage()(image)
    if crop is True:
        img = random_crop_resize_image(image=image, height=size, width=size)
    else:
        train_trans = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize(int(size*1.5)), # Added this since centercrop was cutting into the face
                                        transforms.CenterCrop(size),
                                         transforms.ToTensor()])
        img = train_trans(image)
    if flip:
        img = random_flip_image(img)
    if color_distort:
        img = color_distort_image(img)
    return img


def get_transforms(img_height, img_width, size=None, p=0.5, crop=True, color_jitter=True, flip=True, grayscale=False):
    if not size:
        size = img_height
    aspect_ratio = img_width / img_height
    if crop is True:
        crop = [transforms.transforms.RandomResizedCrop(size=(size, size), scale=(0.08, 1),
                                 ratio=((3. / 4) * aspect_ratio, (4. / 3) * aspect_ratio))]

    else:
        crop = [transforms.Resize((int(size), int(size)))]
    #     crop = [transforms.Resize(int(size * 1.5)),  # Added this since centercrop was cutting into the face
    #             transforms.CenterCrop(size)]

    if color_jitter is True:
        color_jitter = [transforms.RandomApply([transforms.ColorJitter(0.8 * p,
                                                                    0.8 * p,
                                                                    0.8 * p,
                                                                    0.2 * p)], p=0.8),
                                            transforms.RandomGrayscale(p=0.2)]
    else:
        color_jitter = []

    if flip is True:
        flip = [transforms.RandomHorizontalFlip(p=p)]
    else:
        flip = []

    if grayscale is True:
        grayscale = [transforms.RandomGrayscale(p=0.5)]
    else:
        grayscale = []

    to_tensor = [transforms.ToTensor()]

    transform_list = transforms.Compose(crop+grayscale+color_jitter+flip+to_tensor)

    return transform_list
