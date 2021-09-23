import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import sys
import subprocess
import dlib
from imutils import face_utils
import matplotlib.pyplot as plt
from dfd_utils.resnet import get_resnet, name_to_params, StemCIFAR
import itertools

ROOT_PATH = '~/Desktop/codes/DeepFakeDetection/'

#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
#'facenet-pytorch'])
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
#'mtcnn-opencv'])
# !pip install facenet-pytorch
import cv2
from facenet_pytorch import MTCNN
#from mtcnn_cv2 import MTCNN
from skimage import io, transform


def load_chkpt(model_path, chkpt_path, model_head, optimizer=None, head=None):
    chkpt = torch.load(chkpt_path)
    if optimizer:
        optimizer.load_state_dict(chkpt['optimizer'])
    if head:
        head.load_state_dict(chkpt['head'])
    model, _ = get_resnet(*name_to_params(model_path))
    model.fc = model_head
    model.load_state_dict(chkpt['resnet'])
    return model, optimizer, head

def save_chkpt(model, optimizer, epoch, name=None, head=None, dir_name=ROOT_PATH, losses = []):
    if name is None:
        name = f'{model._get_name()}_epoch{epoch}.chkpt'
    else:
        name = name + f'_{epoch}.chkpt'
    if head is None:
        head_state_dict = None
    else:
        head_state_dict = head.state_dict()
    save_dir = os.path.join(dir_name, 'Model')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    path = os.path.join(save_dir, name)
    torch.save({'resnet':model.state_dict(),
                'optimizer':optimizer.state_dict(),
               'head':head_state_dict,
                'losses':losses},path)


def freeze_params(model):
    """ Freeze all parameters aside from FC"""
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    return model


def unfreeze_params(model):
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def load_model(model_path, num_classes):
    model, _ = get_resnet(*name_to_params(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(torch.load(model_path)['resnet'])
    cifar_stem = StemCIFAR(sk_ratio=0, width_multiplier=1)
    model.net[0] = cifar_stem
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_channels,100),
        nn.Linear(100, num_classes))


def get_face_from_img(img, mtcnn):
    """
    img - ndarray
    mtcnn - MTCNN model

    returns image of type ndarry
    """
    sample = Image.fromarray(img, 'RGB')
        
    face = mtcnn.detect(sample)
    face = face.permute(1, 2, 0)
    return face.int().numpy()

def get_mask_from_img(img, mtcnn):
    """
    img - ndarray
    mtcnn - MTCNN model

    returns image of type ndarry
    """
    sample = Image.fromarray(img, 'RGB')
    boxes, probs = mtcnn.detect(sample)
    # check that face was detected:
    if boxes is None:
        return None

    box = boxes[0].tolist() # there should only be 1 face
    
    box = [int(item) for item in box] # conv to int
    # if any(item < 0 for item in box):
    #     return None
    box = [item if item > 0 else 0 for item in box]
    #print('bounding box:', box)

    dim1, dim2, _ = img.shape

    bin_mask = np.zeros((dim1, dim2), dtype=np.uint8)
    bin_mask[box[1]:box[3], box[0]:box[2]] = 255

    #plt.imshow(bin_mask, interpolation='nearest')
    #plt.show()

#     img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
#     print('img shape:', img.shape,' mask shape:', bin_mask.shape)
#     img = cv2.bitwise_and(img, img, mask = bin_mask)
#     plt.imshow(img)
#     plt.show()
    return bin_mask

FF_DATA_PATH = '/media/shirbar/My Passport/FaceForensics/downloads'

def GenerateFaceMasks(path_to_imgs, masks_root_path=None, overwrite=False):
    if masks_root_path is None:
        masks_root_path = FF_DATA_PATH + '/original_sequences/youtube/masks'
    #mtcnn = MTCNN(image_size=ORIGIN_IMG_HEIGHT, margin=ORIGIN_IMG_HEIGHT*0.6, post_process=False)
    mtcnn = MTCNN()
    imgs_wo_mask =[]
    i = 0
    for subdir, dirs, files in os.walk(path_to_imgs):
        for file in files:
            img_path = os.path.join(subdir, file)
            
            mask_folder = os.path.basename(subdir)
            mask_folder = os.path.join(masks_root_path, mask_folder)
            
            suffix = os.path.join(os.path.basename(subdir), file)
            
            mask_path = os.path.join(masks_root_path, suffix)
            
            # check if mask exists
            if overwrite is False:
                if os.path.isfile(mask_path) is True:
                    # skip this mask
                    print('mask exists, for path:', img_path,' skipping')
                    continue
            
            img = io.imread(img_path)
            bin_mask = get_mask_from_img(img, mtcnn)
            if bin_mask is None:
                print('no face was detected at image:', img_path)
                imgs_wo_mask.append(img_path)
                continue
            
            # check if folder exists, if not create it
            if not os.path.exists(mask_folder):
                os.makedirs(mask_folder)

            cv2.imwrite(mask_path, bin_mask)
            
            # test mask occasionally            
            if 2 % 200 == 0:
                mask = cv2.imread(mask_path, 0).astype('uint8')
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                print('mask shape:', mask.shape, ' img shape:',img.shape)
                
                plt.imshow(img)
                plt.show()
                plt.imshow(mask)
                plt.show()

                img = cv2.bitwise_and(img, img, mask=mask)
                plt.imshow(img)
                plt.show()
            i+=1
    
    return imgs_wo_mask


class Blackout():
    def __init__(self, path_to_shape_predictor='/srv/DeepFakeDetection/andrew_atonov_simclr_pytorch/simclr-pytorch/dfd_utils/shape_predictor_68_face_landmarks.dat') -> None:
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(path_to_shape_predictor)

    """
    Receives an image of a face
    Returns image of a face with eyes and mouth blacked out
    If no face is found, return original image
    """
    def blackout_eyes_mouth(self, img):
        face = self.detector(img)
        if len(face) == 0:
            #print('couldnt detect face')
            # TODO add some kind of exception
            return img

        face = face[0] # There should only be a single face in an image
        shape  = self.predictor(image=img, box=face)
        shape = face_utils.shape_to_np(shape)

        mouth = Blackout.get_bounding_rectangle_from_coordinates(shape[range(48,68)])
        r_eye = Blackout.get_bounding_rectangle_from_coordinates(shape[range(36,42)])
        l_eye = Blackout.get_bounding_rectangle_from_coordinates(shape[range(42,48)])

        face_parts = [mouth, r_eye, l_eye]

        for part in face_parts:
            cv2.fillPoly(img, pts=np.int32([np.array(part)]), color=(0,0,0))
        
        return img

    @staticmethod
    def get_bounding_rectangle_from_coordinates(coordinates):
        xs = [point[0] for point in coordinates]
        ys = [point[1] for point in coordinates]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        rec = list(itertools.product(*[[min_x, max_x], [max_y,min_y]]))
        # TODO do something more generic when you're not tired
        rec[2], rec[3] = rec[3], rec[2] # This orders it so rec[2] is the closer point to rec[1]

        return rec
        
        
