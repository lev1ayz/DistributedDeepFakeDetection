#from _typeshed import NoneType
import importlib
from torch.functional import Tensor
from torchvision import transforms
from torchvision.transforms.functional import crop
from dfd_utils.datasets import get_transforms
import torch
import numpy as np
from torch.utils.data import Dataset
import os
import cv2
import re
from dfd_utils import ImageAugmentation
importlib.reload(ImageAugmentation)
from dfd_utils.ImageAugmentation import get_transforms
from dfd_utils.utils import GenerateFaceMasks, Blackout

from PIL import Image


# Custom Dataset class for loading images   
"""
DONT CHANGE THE LABEL ASSIGMENT
"""
REAL_LABEL = 1
FAKE_LABEL = 0

CROPPED_SIZE = 224

"""
Note: the impl. includes seperation to fake and real lists (adfdfnd not just one list) so it would be easier
to impl. a superised learnin dataset if we ant it

Note: load_two functionality is not impl.
"""
class FaceForensicsDataset(Dataset):
    def __init__(self, root_dir, transform=None, mtcnn=None, load_deepfakes=False, load_face2face=False,
                 load_neural_textures=False, masking_transforms=True):
        self.root_dir = root_dir
        
        # Real images
        self.real_img_actors_path           = self.root_dir + '/original_sequences/actors/c23/images'
        self.real_img_youtube_path          = self.root_dir + '/original_sequences/youtube/c23/images'
        
        # Real masks
        self.real_masks_actors_path         = None # no masks so far
        self.real_masks_youtube_path        = self.root_dir + '/original_sequences/youtube/masks'
        
        # Fake images
        self.fake_img_actors_path           = self.root_dir + '/manipulated_sequences/DeepFakeDetection/c23/images'
        self.fake_img_youtube_path          = self.root_dir + '/manipulated_sequences/Deepfakes/c23/images'
        self.fake_img_face2face_path        = self.root_dir + '/manipulated_sequences/Face2Face/c23/images'
        self.fake_img_nerualtextures_path   = self.root_dir + '/manipulated_sequences/NeuralTextures/c23/images'
        
        # Fake Masks
        self.fake_masks_actors_path         = self.root_dir + '/manipulated_sequences/DeepFakeDetection/masks/images'
        self.fake_masks_youtube_path        = self.root_dir + '/manipulated_sequences/Deepfakes/masks/images'
        self.fake_masks_face2face_path      = self.root_dir + '/manipulated_sequences/Face2Face/masks/images'
        self.fake_masks_nerualtextures_path = self.root_dir + '/manipulated_sequences/NeuralTextures/masks/c23/images'
        
        self.transform = transform
        
        self.targets = []
        
        self.real_img_paths = []
        self.real_masks_paths = []
        
        self.fake_img_paths = []
        self.fake_img_masks_paths = []
        
        self.img_paths = []  # These will be used in __getitem__
        self.mask_paths = [] #
        
        self.mtcnn = mtcnn
        self.load_deepfakes = load_deepfakes
        self.load_face2face = load_face2face
        self.load_neural_textures = load_neural_textures
        self.masking_transforms = masking_transforms

        self.blackout = Blackout()

        # Load real imgs
        self.real_img_paths, self.real_masks_paths = self.load_imgs_and_masks(self.real_img_youtube_path,
                                                                            self.real_masks_youtube_path)
        # Load fake imgs
        if self.load_deepfakes is True:
            self.fake_img_paths, self.fake_img_masks_paths = self.load_imgs_and_masks(
                                                        self.fake_img_youtube_path,
                                                        self.fake_masks_youtube_path)
        
        if self.load_face2face is True:             
            face2face_imgs, face2face_masks = self.load_imgs_and_masks(
                                                                    self.fake_img_face2face_path,
                                                                    self.fake_masks_face2face_path)
            # append to the face2face to the fake imgs
            self.fake_img_paths = self.fake_img_paths + face2face_imgs
            self.fake_img_masks_paths = self.fake_img_masks_paths + face2face_masks

        if self.load_neural_textures is True:
            neural_textures_imgs, neural_textures_masks = self.load_imgs_and_masks(
                                                                                self.fake_img_nerualtextures_path,
                                                                                self.fake_masks_nerualtextures_path)
            # append neural textures to the fake imgs
            self.fake_img_paths = self.fake_img_paths + neural_textures_imgs
            self.fake_img_masks_paths = self.fake_img_masks_paths + neural_textures_masks

        print('real imgs len:',len(self.real_img_paths))
        print('masks len:',len(self.real_masks_paths))
        print('fakes imgs len:',len(self.fake_img_paths))
        print('fakes masks len:',len(self.fake_img_masks_paths))
        
        # Transfer real and fakes to a single list
        self.img_paths = self.real_img_paths + self.fake_img_paths
        self.mask_paths = self.real_masks_paths + self.fake_img_masks_paths
        
        # Create appropriate labels
        self.targets = np.ones_like(self.real_img_paths, dtype=np.int8)
        self.targets = np.concatenate((self.targets ,np.zeros(len(self.fake_img_paths), dtype=np.int8)))
        self.targets = list(self.targets)
        
        print('final imgs len:', len(self.img_paths))
        print('final masks len:',len(self.mask_paths))
        
        print('asserting order')
        self.assert_order(self.img_paths, self.mask_paths)

    def __len__(self):
        return len(self.img_paths)
  
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            print('no mask at path:', mask_path)

        if mask is None or all(item == False for item in mask.any(1)) == True:
            print('invalid mask')
            path_to_imgs = os.path.dirname(img_path)
            path_to_masks = os.path.dirname(os.path.dirname(mask_path)) # avoid creating a subfolder within mask folder
            print('remasking:', path_to_imgs, ' to dest:', path_to_masks)
            GenerateFaceMasks(path_to_imgs, masks_root_path=path_to_masks, overwrite=True)
            mask = cv2.imread(mask_path, 0)

        if mask is None or all(item == False for item in mask.any(1)) == True or all(item == False for item in mask.any(0)) == True:
            # if even after remasking, theres no mask, use whole picture
            pass
        else:
            img = img[np.ix_(mask.any(1), mask.any(0))]


        if self.transform is None:
            img_width, img_height, _ = img.shape
            self.transform = get_transforms(img_width, img_height, CROPPED_SIZE, crop=False, color_jitter=False, grayscale=True)

        if self.masking_transforms:
            img1 = self.blackout.random_blackout_eyes_mouth(img) # Blackout eyes and mouth
            img2 = self.blackout.random_blackout_half_of_img(img) # Blackout random half of img

        elif not (isinstance(img, type(torch.TensorType))) or not (isinstance(img, type(Image))):
            img1 = transforms.ToPILImage()(img)
            img2 = transforms.ToPILImage()(img)
        else:
            img1 = img
            img2 = img

        img1 = self.transform(img1)
        img2 = self.transform(img2)

                
        return img1, img2, self.targets[idx]


    """
    Removes extra paths from self.images and self.masks so that theres an equal amount of real and fakes
    This function takes advantage of the fact the the 1st half of self.imgs are real images and the 2nd part are fake images
    """
    def equalize_real_fakes(self):
        real_len = len(self.real_img_paths)
        fake_len = len(self.fake_img_paths)
        print('before eq: real:', real_len, ' fake:', fake_len, ' total:', len(self.img_paths))

        diff = fake_len - real_len
        # if diff is > 0 then cutout part from real_len->real_len+diff
        # else cutout from real_len ->real_len+diff (becuase diff < 0  it will cut into the real images)
        if diff > 0:
            del self.img_paths[real_len:real_len+diff]
            del self.mask_paths[real_len:real_len+diff]
            del self.targets[real_len:real_len+diff]
        else:
            del self.img_paths[real_len-diff:real_len]
            del self.mask_paths[real_len-diff:real_len]
            del self.targets[real_len-diff:real_len]

        print('after eq:', 'total:', len(self.img_paths))

        return

    def load_imgs_and_masks(self, imgs_path, masks_path):
        img_list = sorted(self.load_img_paths_to_list(imgs_path))
        masks_list = sorted(self.load_img_paths_to_list(masks_path))

        img_list, masks_list = self.get_ordered_img_masks_lists(img_list, masks_list)

        return img_list, masks_list

    def get_ordered_img_masks_lists(self, img_list, mask_list):
        img_suffix_set = set([FaceForensicsDataset.get_suffix_from_path(img) for img in img_list])
        mask_suffix_set = set([FaceForensicsDataset.get_suffix_from_path(mask) for mask in mask_list])

        orderd_img_list = sorted([img for img in img_list if (FaceForensicsDataset.get_suffix_from_path(img) in mask_suffix_set)])
        ordered_mask_list = sorted([mask for mask in mask_list if (FaceForensicsDataset.get_suffix_from_path(mask) in img_suffix_set)])

        return orderd_img_list, ordered_mask_list

    @staticmethod
    def get_suffix_from_path(path):
        file = os.path.basename(path)
        folder = os.path.basename(os.path.dirname(path))
        return folder + '/' + file
    
    def assert_order(self, img_list, mask_list): # Not used, order is asserted in transfer_omg_paths..
        assert len(img_list) == len(mask_list), "img list len different from mask list len!"
        
        for idx, img in enumerate(img_list):
            assert FaceForensicsDataset.get_suffix_from_path(img) == FaceForensicsDataset.get_suffix_from_path(mask_list[idx]) , "mask incompatible with image!"
        
        print('assertion passed!')
        return
                                                                                  
    def load_img_paths_to_list(self,path):
        lst = []
        for subdir, dirs, files in os.walk(path, topdown=True):
            for file in files:
                if FaceForensicsDataset.check_img_valid(file) is True:
                    lst.append(os.path.join(subdir, file))
                else:
                    print('invalid img:', file)
        return lst
    
    @staticmethod
    def check_img_valid(img_path):
        if img_path.endswith('.png') or img_path.endswith('.jpg'):
            return True
        return False


class FaceForensicsForClassificationDataset(FaceForensicsDataset):
    def __init__(self, root_dir, transform, mtcnn=None, load_deepfakes=True, load_face2face=True):
        super().__init__(root_dir, transform=transform, mtcnn=mtcnn, load_deepfakes=load_deepfakes,
                         load_face2face=load_face2face)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # maybe dont to save loading time
        mask = cv2.imread(mask_path, 0)

        if all(item == False for item in mask.any(1)) == True:
            print('invalid mask')
            path_to_imgs = os.path.dirname(img_path)
            path_to_masks = os.path.dirname(os.path.dirname(mask_path)) # avoid creating a subfolder within mask folder
            print('remasking:', path_to_imgs, ' to dest:', path_to_masks)
            GenerateFaceMasks(path_to_imgs, masks_root_path=path_to_masks, overwrite=True)
            mask = cv2.imread(mask_path, 0)

        if all(item == False for item in mask.any(1)) == True or all(item == False for item in mask.any(0)) == True:
            # if even after remasking, no mask use whole picture
            pass
        else:
            img = img[np.ix_(mask.any(1), mask.any(0))]

        width, height, _ = img.shape

        if self.transform is None:
            self.transform = get_transforms(width, height, CROPPED_SIZE, crop=False, color_jitter=False, flip=True)

        if not (isinstance(img, type(torch.TensorType)) or isinstance(img, type(Image))):
            img = transforms.ToPILImage()(img)
        
        img = self.transform(img)

        return img, self.targets[idx]
        