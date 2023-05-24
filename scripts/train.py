import land_cover_models as lcm
import torch
import torch.nn as nn
from torchvision import transforms, utils
from fmix import combine_masks
from my_transformations import RandomCrop, ToTensor, extra_transforms
from models import SiameseNetwork
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# ------------------- CONSTANTS ------------------- #

MEAN=[ 94.7450, 102.9471,  96.9884]
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
# Define transforms
transform=transforms.Compose([RandomCrop(CROP_SIZE),
                              ToTensor(),
                              transforms.Normalize(mean=MEAN, std=STD)
                              ])

# TODO: Dataset train, val, test split
# Define dataset
ds = lcm.DataSetPatches(im_dir=dir_test_im_patches, mask_dir=dir_test_mask_patches, 
                                    mask_suffix=mask_suffix_test_ds, mask_dir_name=mask_dir_name_test,
                                #   list_tile_names=dict_tile_names_sample['test'],
                                    list_tile_patches_use=None,
                                    shuffle_order_patches=True, relabel_masks=True,
                                    random_transform_data=True,
                                    subsample_patches=False, # frac_subsample=0.1,
                                    path_mapping_dict=path_mapping_dict, transform=transform
                                    )

# Define dataloader
trainloader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, pin_memory=True,
                                          shuffle=True, num_workers=2)



torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



# Train the model

model = SiameseNetwork().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=LR)

scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
criterion = nn.BCELoss()

for epoch in range(1, EPOCHS + 1):
    
    model.train()

    # Training loop
    for batch_idx, batch in enumerate(trainloader):
        
        # Create the artificial augmented image and get the change mask
        original_image, augmented_image, change_mask = combine_masks(batch, "continuous_fmix")
        
        original_image_transformed = extra_transforms(original_image)
        augmented_image_transformed = extra_transforms(augmented_image)
        
        # Send to CUDA device
        original_image = original_image.to(device)
        augmented_image = augmented_image.to(device)
        change_mask = change_mask.to(device)
        
        optimizer.zero_grad()
        outputs = model(original_image, augmented_image).squeeze()
        loss = criterion(outputs, change_mask)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(original_image), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item()))
    
        scheduler.step()
    
    # TODO: Validation loop