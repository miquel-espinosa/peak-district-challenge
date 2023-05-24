import land_cover_models as lcm
import torch
from fmix import combine_masks
from matplotlib import pyplot as plt
import math

ROOT_PATH = '/home/s2254242/PHD/ATI/peak-district-challenge'

dir_test_im_patches = f'{ROOT_PATH}/data/images_detailed_annotation/'
dir_test_mask_patches = f'{ROOT_PATH}/data/masks_detailed_annotation/'

mask_suffix_test_ds = '_lc_2022_detailed_mask.npy'
mask_dir_name_test = ''

path_mapping_dict=f'{ROOT_PATH}/content/label_mapping_dicts/label_mapping_dict__main_categories__2023-04-20-1541.pkl'

ds = lcm.DataSetPatches(im_dir=dir_test_im_patches, mask_dir=dir_test_mask_patches, 
                                    mask_suffix=mask_suffix_test_ds, mask_dir_name=mask_dir_name_test,
                                #   list_tile_names=dict_tile_names_sample['test'],
                                    list_tile_patches_use=None,
                                    shuffle_order_patches=True, relabel_masks=True,
                                    subsample_patches=False, # frac_subsample=0.1,
                                    path_mapping_dict=path_mapping_dict
                                    )

trainloader = torch.utils.data.DataLoader(ds, batch_size=1)
summ = torch.zeros((3))
summ_x2 = torch.zeros((3))

min = torch.tensor([math.inf, math.inf, math.inf])
max = torch.tensor([-math.inf, -math.inf, -math.inf])

for img, _ in trainloader:
    for i in range(0, len(summ)):
        min[i] = torch.min(img[0,i,:,:].min(), min[i])
        max[i] = torch.max(img[0,i,:,:].min(), max[i])
        summ[i] += img[0,i,:,:].sum()
        summ_x2[i] += (img[0,i,:,:] ** 2).sum()

example_img = ds.__getitem__(0)[0]
total_pixels = example_img.shape[-1] * example_img.shape[-2] * len(ds)

mean = summ/total_pixels


stdev = torch.sqrt((summ_x2 / total_pixels) - (mean * mean)) 
print("mean", mean)
print('std', stdev)

print('min', min)
print('max', max)