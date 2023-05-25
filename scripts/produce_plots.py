import matplotlib.pyplot as plt

import fmix
import numpy as np

def combine_masks_predictions(x, method=None):
    img, mask = x
    original = img  # .deepcopy() TODO: do we want rectangular edges?
    permute = torch.randperm(img.shape[0])
    class_change = mask != mask[permute]
    artificial_mask = fmix.get_mask(method, img)

    # print(artificial_mask.shape) # TODO: get different masks per image in the batch
    # Currently we have a single mask for the entire batch

    final_mask = torch.logical_and(torch.tensor(artificial_mask), class_change)
    final_mask = final_mask * artificial_mask
    modified_image = img * artificial_mask + img[permute] * (1 - artificial_mask)

    return original, modified_image, final_mask, img[permute], artificial_mask, class_change, permute


import land_cover_models as lcm
import torch
import torch.nn as nns
from my_transformations import RandomCrop, extra_transforms
import argparse
from unet import UNet

# ROOT_PATH = '/home/s2254242/PHD/ATI/peak-district-challenge'
ROOT_PATH = '/shared/miguel/peak-district-challenge'
dir_test_im_patches = f'{ROOT_PATH}/data/images_detailed_annotation/'
dir_test_mask_patches = f'{ROOT_PATH}/data/masks_detailed_annotation/'
path_mapping_dict = f'{ROOT_PATH}/content/label_mapping_dicts/label_mapping_dict__main_categories__2023-04-20-1541.pkl'

mask_suffix_test_ds = '_lc_2022_detailed_mask.npy'
mask_dir_name_test = ''

MEAN=[94.7450, 102.9471, 96.9884]
STD=[31.5119, 24.6287, 19.7533]

TRAIN_TEST_RATIO = 0.8

CROP_SIZE = 128 # RandomCrop transform
BATCH_SIZE = 10
device = 'cuda:0'

# ------------------- TRANSFORMS, DATASETS, DATALOADER ------------------- #

# TODO: Dataset train, val, test split
# Define dataset
ds = lcm.DataSetPatches(im_dir=dir_test_im_patches, mask_dir=dir_test_mask_patches,
                        mask_suffix=mask_suffix_test_ds, mask_dir_name=mask_dir_name_test,
                        #   list_tile_names=dict_tile_names_sample['test'],
                        list_tile_patches_use=None,
                        shuffle_order_patches=True, relabel_masks=True,
                        random_transform_data=True,
                        subsample_patches=False,  # frac_subsample=0.1,
                        path_mapping_dict=path_mapping_dict,
                        random_crop=RandomCrop(CROP_SIZE),
                        mean=MEAN, std=STD)

length = len(ds)

# Do the train-test split
perm = torch.randperm(len(ds))

train_ds = torch.utils.data.Subset(ds, perm[:int(TRAIN_TEST_RATIO * length)])
val_ds = torch.utils.data.Subset(ds, perm[int(TRAIN_TEST_RATIO * length):])

# Define dataloaders
trainloader = torch.utils.data.DataLoader(train_ds, batch_size=8, pin_memory=True,
                                          shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(val_ds, batch_size=8, pin_memory=True,
                                        shuffle=True, num_workers=2)

# model = SiameseNetwork(output_dim=(1,CROP_SIZE*CROP_SIZE)).to(device)
# model = SNUNet_ECAM(3, 1).to(device)
model = UNet(n_channels=3, n_classes=1).to(device)
model.load_state_dict(torch.load('/shared/miguel/peak-district-challenge/results_inclass_mixing/model_40.pth'))
model.eval()

it = iter(valloader)
batch = next(it)

# change_mask = change_mask.float()

color = [tuple(np.random.choice(range(256), size=3)) for i in range(50)]
def get_fixed_colours_image(mask):
    finalmask = np.zeros((mask.shape[0], mask.shape[1], 3))
    for label in range(len(color)):
        finalmask[mask == label] = color[label]
    return finalmask/255

i=0

fig3 = plt.figure(constrained_layout=True)
gs = fig3.add_gridspec(2, 4)

# for i in range(10):
#     original_image, augmented_image, change_mask, mixing_img, fouriermask, class_change, permute = combine_masks_predictions(batch, "continuous_fmix")
#     f3_ax1 = fig3.add_subplot(gs[0, 0])
#     f3_ax1.set_title('Source image 1')
#     f3_ax1.imshow(original_image[0].permute(1,2,0))
#     f3_ax1.axis('off') 
#     f3_ax2 = fig3.add_subplot(gs[0, 1])
#     f3_ax2.set_title('Source image 2')
#     f3_ax2.imshow(mixing_img[0].permute(1,2,0))
#     f3_ax2.axis('off')
#     f3_ax3 = fig3.add_subplot(gs[:, 2])
#     f3_ax3.set_title('Sampled mask')
#     f3_ax3.imshow(fouriermask[0])
#     f3_ax3.axis('off')
#     f3_ax1 = fig3.add_subplot(gs[0, 3])
#     f3_ax1.set_title('Changed image')
#     f3_ax1.imshow(augmented_image[0].permute(1,2,0))
#     f3_ax1.axis('off')
#     f3_ax4 = fig3.add_subplot(gs[1, 0])
#     f3_ax4.set_title('Land cover 1')
#     f3_ax4.imshow(get_fixed_colours_image(batch[1][0]))
#     f3_ax4.axis('off')
#     f3_ax5 = fig3.add_subplot(gs[1, 1])
#     f3_ax5.set_title('Land cover 2')
#     f3_ax5.imshow(get_fixed_colours_image(batch[1][permute][0]))
#     f3_ax5.axis('off')
#     f3_ax5 = fig3.add_subplot(gs[1, 3])
#     f3_ax5.set_title('Change map')
#     f3_ax5.imshow(get_fixed_colours_image(change_mask[0]))
#     f3_ax5.axis('off')
#     plt.savefig(f'mask_generation_example_{i}.png')


fig3 = plt.figure(constrained_layout=True)
gs = fig3.add_gridspec(4, 4)
for i in range(4):
        
    for j in range(4):
        original_image, augmented_image, change_mask, mixing_img, fouriermask, class_change, permute = combine_masks_predictions(batch, "continuous_fmix")

        f3_ax1 = fig3.add_subplot(gs[0, j])
        # f3_ax1.set_title('Source image 1')
        f3_ax1.imshow(original_image[j].permute(1,2,0))
        f3_ax1.axis('off') 

        f3_ax1 = fig3.add_subplot(gs[1, j])
        # f3_ax1.set_title('Source image 1')
        f3_ax1.imshow(original_image[permute][j].permute(1,2,0))
        f3_ax1.axis('off') 

        f3_ax1 = fig3.add_subplot(gs[2, j])
        # f3_ax1.set_title('Source image 1')
        f3_ax1.imshow(fouriermask[0])
        f3_ax1.axis('off') 

        f3_ax1 = fig3.add_subplot(gs[3, j])
        # f3_ax1.set_title('Source image 1')
        f3_ax1.imshow(augmented_image[j].permute(1,2,0))
        f3_ax1.axis('off') 


    # f3_ax2 = fig3.add_subplot(gs[0, 1])
    # # f3_ax2.set_title('Source image 2')
    # f3_ax2.imshow(mixing_img[0].permute(1,2,0))
    # f3_ax2.axis('off')
    # f3_ax3 = fig3.add_subplot(gs[:, 2])
    # # f3_ax3.set_title('Sampled mask')
    # f3_ax3.imshow(fouriermask[0])
    # f3_ax3.axis('off')
    # f3_ax1 = fig3.add_subplot(gs[0, 3])
    # # f3_ax1.set_title('Changed image')
    # f3_ax1.imshow(augmented_image[0].permute(1,2,0))
    # f3_ax1.axis('off')
    # f3_ax4 = fig3.add_subplot(gs[1, 0])
    # # f3_ax4.set_title('Land cover 1')
    # f3_ax4.imshow(get_fixed_colours_image(batch[1][0]))
    # f3_ax4.axis('off')
    # f3_ax5 = fig3.add_subplot(gs[1, 1])
    # # f3_ax5.set_title('Land cover 2')
    # f3_ax5.imshow(get_fixed_colours_image(batch[1][permute][0]))
    # f3_ax5.axis('off')
    # f3_ax5 = fig3.add_subplot(gs[1, 3])
    # f3_ax5.set_title('Change map')
    # f3_ax5.imshow(get_fixed_colours_image(change_mask[0]))
    # f3_ax5.axis('off')
plt.savefig(f'generated_augs_{5}.png')