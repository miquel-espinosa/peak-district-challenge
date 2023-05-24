import land_cover_models as lcm
import torch
import torch.nn as nn
from torchvision import transforms, utils
from fmix import combine_masks
from my_transformations import RandomCrop, extra_transforms
from models import SiameseNetwork
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from skimage import color
from skimage import io


# ------------------- CONSTANTS ------------------- #

MEAN=[94.7450, 102.9471, 96.9884]
STD=[31.5119, 24.6287, 19.7533]

LOG_INTERVAL = 10

CROP_SIZE = 128 # RandomCrop transform
BATCH_SIZE = 8
LR = 0.001
EPOCHS=50
SEED=42

# ------------------- DIRECTORY PATHS ------------------- #

ROOT_PATH = '/home/s2254242/PHD/ATI/peak-district-challenge'
dir_test_im_patches = f'{ROOT_PATH}/data/images_detailed_annotation/'
dir_test_mask_patches = f'{ROOT_PATH}/data/masks_detailed_annotation/'
path_mapping_dict=f'{ROOT_PATH}/content/label_mapping_dicts/label_mapping_dict__main_categories__2023-04-20-1541.pkl'

mask_suffix_test_ds = '_lc_2022_detailed_mask.npy'
mask_dir_name_test = ''


# ------------------- TRANSFORMS, DATASETS, DATALOADER ------------------- #

# TODO: Dataset train, val, test split
# Define dataset
ds = lcm.DataSetPatches(im_dir=dir_test_im_patches, mask_dir=dir_test_mask_patches, 
                                    mask_suffix=mask_suffix_test_ds, mask_dir_name=mask_dir_name_test,
                                #   list_tile_names=dict_tile_names_sample['test'],
                                    list_tile_patches_use=None,
                                    shuffle_order_patches=True, relabel_masks=True,
                                    random_transform_data=True,
                                    subsample_patches=False, # frac_subsample=0.1,
                                    path_mapping_dict=path_mapping_dict,
                                    random_crop=RandomCrop(CROP_SIZE),
                                    mean=MEAN, std=STD)

# Define dataloader
trainloader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, pin_memory=True,
                                          shuffle=True, num_workers=2)


CHOOSE_BATCH_IDX = 2

iterator = iter(trainloader)
i = 0
found = False
while found == False:
    
    batch = next(iterator)
    
    if i == CHOOSE_BATCH_IDX and found == False:
        found = True
        
        original_image, augmented_image, change_mask, mixing_img, fouriermask = combine_masks(batch, "continuous_fmix")
        original_image_transformed = extra_transforms(original_image)
        augmented_image_transformed = extra_transforms(augmented_image)
        
        # Plot the original image and mask
        fig, ax = plt.subplots(2,3, figsize=(10,10))
        ax[0][0].set_title('Original image 1')
        ax[0][0].imshow(original_image[i].permute(1,2,0))#.type(torch.uint8))#, vmin=0, vmax=1)
        ax[0][1].set_title('Mixing with image 2')
        ax[0][1].imshow(mixing_img[i].permute(1,2,0))#.type(torch.uint8))#, vmin=0, vmax=1)
        ax[0][2].set_title('Fourier mask')
        ax[0][2].imshow(fouriermask[0])
        # ax[0][1].imshow(batch[1][i])
        # io.imshow(color.label2rgb(batch[1][i],batch[0][i].permute(1,2,0)))
        # ax[0][1].set_title('Original mask 1')
        
        ax[1][0].set_title('Augmentations image 1')
        ax[1][0].imshow(original_image_transformed[i].permute(1,2,0))
        ax[1][1].set_title('Augmentations image 2')
        ax[1][1].imshow(augmented_image_transformed[i].permute(1,2,0))
        ax[1][2].set_title('Change mask between 1 and 2')
        ax[1][2].imshow(change_mask[i])
        
        
        plt.show()
        
    i = i + 1
