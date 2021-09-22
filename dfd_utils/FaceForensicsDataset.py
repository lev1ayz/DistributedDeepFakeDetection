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
from dfd_utils import ImageAugmentation
importlib.reload(ImageAugmentation)
from dfd_utils.ImageAugmentation import get_transforms
from dfd_utils.utils import GenerateFaceMasks, Blackout

from PIL import Image

#import sys
#sys.path.insert(0, '/srv/DeepFakeDetection/andrew_atonov_simclr_pytorch/simclr-pytorch/dfd_utils/')
#from utils import GenerateFaceMasks

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
    def __init__(self, root_dir, transform=None, mtcnn=None, load_deepfakes=False, load_face2face=False):
        self.root_dir = root_dir
        
        self.real_img_actors_path = self.root_dir + '/original_sequences/actors/c23/images'
        self.real_img_youtube_path = self.root_dir + '/original_sequences/youtube/c23/images'
        
        self.real_masks_actors_path = None # no masks so far
        self.real_masks_youtube_path = self.root_dir + '/original_sequences/youtube/masks'
        
        self.fake_img_actors_path = self.root_dir + '/manipulated_sequences/DeepFakeDetection/c23/images'
        self.fake_img_youtube_path = self.root_dir + '/manipulated_sequences/Deepfakes/c23/images'
        self.fake_img_face2face_path = self.root_dir + '/manipulated_sequences/Face2Face/c23/images'
        
        self.fake_masks_actors_path = self.root_dir + '/manipulated_sequences/DeepFakeDetection/masks/images'
        self.fake_masks_youtube_path = self.root_dir + '/manipulated_sequences/Deepfakes/masks/images'
        self.fake_masks_face2face_path = self.root_dir + '/manipulated_sequences/Face2Face/masks/images'
        
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
                
        # load youtube part
        self.real_img_paths = sorted(self.load_img_paths_to_list(self.real_img_youtube_path, self.real_img_paths))
        self.real_masks_paths = sorted(self.load_img_paths_to_list(self.real_masks_youtube_path, self.real_masks_paths))
        
        if self.load_deepfakes is True:
            self.fake_img_paths = sorted(self.load_img_paths_to_list(self.fake_img_youtube_path, self.fake_img_paths))
            self.fake_img_masks_paths = sorted(self.load_img_paths_to_list(
                                                                    self.fake_masks_youtube_path,
                                                                    self.fake_img_masks_paths))
        
        if self.load_face2face is True:             
            self.fake_img_paths = sorted(self.load_img_paths_to_list(
                                                        self.fake_img_face2face_path,
                                                        self.fake_img_paths))
            self.fake_img_masks_paths = sorted(self.load_img_paths_to_list(
                                                                    self.fake_masks_face2face_path,
                                                                    self.fake_img_masks_paths))
        
        print('real imgs len:',len(self.real_img_paths))
        print('masks len:',len(self.real_masks_paths))
        print('fakes imgs len:',len(self.fake_img_paths))
        print('fakes masks len:',len(self.fake_img_masks_paths))
        
        
        # TODO add loading actors part
        
        
        # transfer to one list
        print('ordering real imgs and masks...')
        self.transfer_img_paths_and_mask_paths_to_new_lists(self,
                                                            self.real_img_paths,
                                                            self.real_masks_paths,
                                                            self.img_paths,
                                                            self.mask_paths)
        real_targets = np.ones_like(self.img_paths, dtype=np.int8)
        print('real imgs and masks in order!')
        
        if self.load_deepfakes is True or self.load_face2face is True:
            print('ordering fake imgs and masks....')
            tmp_img_list = []
            tmp_mask_list = []
            self.transfer_img_paths_and_mask_paths_to_new_lists(self,
                                                                self.fake_img_paths,
                                                                self.fake_img_masks_paths,
                                                                tmp_img_list,
                                                                tmp_mask_list)
            fake_targets = np.zeros(len(tmp_img_list), dtype=np.int8)
            self.img_paths = self.img_paths + tmp_img_list
            self.mask_paths = self.mask_paths + tmp_mask_list
            print('fake imgs and masks in order!')
        else:
            fake_targets = []
        
        self.targets = np.concatenate((real_targets,fake_targets))
        self.targets = list(self.targets)
        self.targets = torch.tensor(self.targets)
        
        print('final imgs len:', len(self.img_paths))
        print('final masks len:',len(self.mask_paths))
        
        print('asserting order')
        self.assert_order(self.img_paths, self.mask_paths)

        self.blackout = Blackout()
        

        
    def __len__(self):
        return len(self.img_paths)
  
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # maybe dont to save loading time
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            print('mask path:', mask_path)

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
            #img = cv2.bitwise_and(img, img, mask = mask)
            img = img[np.ix_(mask.any(1), mask.any(0))]

        # Blackout eyes and mouth
        img = self.blackout.blackout_eyes_mouth(img)

        img_width, img_height, _ = img.shape
        
        if self.transform is None:
            self.transform = get_transforms(img_width, img_height, CROPPED_SIZE)

        if not (isinstance(img, type(torch.TensorType))) or not (isinstance(img, type(Image))):
            #print(img.shape)
            #print(img_path)
            #img = transforms.ToPILImage()(img)
            #img = transforms.ToTensor()(img)
            pass
        
        #img1 = self.transform(img)
        #img2 = self.transform(img)
        img1 = img2 = img
        
        # img1 = augment_image(img,size=CROPPED_SIZE, crop=False, flip=False, color_distort=False)
        # img2 = augment_image(img,size=CROPPED_SIZE, crop=False, flip=False, color_distort=False)
                
        return img1, img2, self.targets[idx]
    
    """
    each mask should have an img and i assume initially the list should be somewhat ordered
    """
    @staticmethod
    def transfer_img_paths_and_mask_paths_to_new_lists(self,
                                                       img_list,
                                                       mask_list,
                                                       new_img_list,
                                                       new_mask_list):
        img_div = 0
        for idx,mask in enumerate(mask_list):
            # only transfer if mask has a face
            img = img_list[idx + img_div]
            skip = False
            while self.get_suffix_from_path(mask) != self.get_suffix_from_path(img):
                img_div+=1
                if img_div + idx >= len(img_list):
                    img_div = 0
                    skip = True
                    break
                img = img_list[idx + img_div]
            
            if skip is True:
                continue
                
            new_img_list.append(img)
            new_mask_list.append(mask)
                
        return new_img_list, new_mask_list
    
    @staticmethod
    def get_suffix_from_path(path):
        file = os.path.basename(path)
        folder = os.path.basename(os.path.dirname(path))
        return folder + '/' + file
    
    def assert_order(self, img_list, mask_list): # Not used, order is asserted in transfer_omg_paths..
        assert len(img_list) == len(mask_list), "img list len different from mask list len!"
        
        for idx, img in enumerate(img_list):
            assert self.get_suffix_from_path(img) == self.get_suffix_from_path(mask_list[idx]) , "mask incompatible with image!"
        
        print('assertion passed!')
        return
                                                                                  
    def load_img_paths_to_list(self, path, lst):
        for subdir, dirs, files in os.walk(path, topdown=True):
            for file in files:
                if self.check_img_valid(file) is True:
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
            #GenerateFaceMasks(path_to_imgs, overwrite=True)
            GenerateFaceMasks(path_to_imgs, masks_root_path=path_to_masks, overwrite=True)
            mask = cv2.imread(mask_path, 0)

        if all(item == False for item in mask.any(1)) == True or all(item == False for item in mask.any(0)) == True:
            # if even after remasking, no mask use whole picture
            pass
        else:
            #img = cv2.bitwise_and(img, img, mask = mask)
            img = img[np.ix_(mask.any(1), mask.any(0))]

        #print('img type:', type(img))
        #print('img shape', img.shape)
        width, height, _ = img.shape
        #print('width :', width, ' height:', height)

        if self.transform is None:
            self.transform = get_transforms(width, height, CROPPED_SIZE, crop=False, color_jitter=False, flip=False)

        if not (isinstance(img, type(torch.TensorType)) or isinstance(img, type(Image))):
            #print(img.shape)
            #print(img_path)
            img = transforms.ToPILImage()(img)
        
        img = self.transform(img)

        return img, self.targets[idx]
        