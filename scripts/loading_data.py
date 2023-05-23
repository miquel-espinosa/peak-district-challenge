import land_cover_models as lcm
import torch
# import glob
# from glob import glob


ROOT_PATH = '/home/s2254242/PHD/ATI/peak-district-challenge/'

dir_test_im_patches = 'data/images_detailed_annotation/*.npy'
dir_test_mask_patches = 'data/masks_detailed_annotation/*.npy'

mask_suffix_test_ds = '_lc_2022_detailed_mask.npy'
mask_dir_name_test = ''

path_mapping_dict=f'{ROOT_PATH}/content/label_mapping_dicts/label_mapping_dict__main_categories__2022-11-17-1512.pkl'

ds = lcm.DataSetPatches(im_dir=dir_test_im_patches, mask_dir=dir_test_mask_patches, 
                                    mask_suffix=mask_suffix_test_ds, mask_dir_name=mask_dir_name_test,
                                #   list_tile_names=dict_tile_names_sample['test'],
                                    list_tile_patches_use=None,
                                    shuffle_order_patches=True, relabel_masks=True,
                                    subsample_patches=False, # frac_subsample=0.1,
                                    path_mapping_dict=path_mapping_dict
                                    )


trainloader = torch.utils.data.DataLoader(ds)

