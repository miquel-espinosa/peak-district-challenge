import numpy 
import matplotlib.pyplot as plt
import imageio 
import numpy as np 
import land_cover_models as lcm
import torch
import torch.nn as nn
from torchvision import transforms, utils
from fmix import combine_masks_prediction
from my_transformations import RandomCrop, extra_transforms
from models import SiameseNetwork
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import matplotlib.pyplot as plt
import argparse
from unet import UNet


parser = argparse.ArgumentParser(description='training stuff')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--optim', type=str, default='adam',
                    choices=['adamw', 'sgd', 'adadelta'])
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save-model-path', type=str, default='results')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
args = parser.parse_args()


def compute_loss_and_acc(loader, model, criterion, subset):
    loss = 0
    correct = 0
    for batch in loader:
        images_1, images_2, targets = combine_masks(batch, "continuous_fmix")
        images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
        outputs = model(images_1.float(), images_2.float()).squeeze()
        loss += criterion(outputs, targets).sum().item()  # sum up batch loss
        pred = torch.where(outputs > 0.5, 1, 0)  # get the index of the max log-probability
        targets = torch.where(targets > 0.5, 1, 0)
        correct += pred.eq(targets.view_as(pred)).sum().item()
    loss /= len(loader.dataset)
    correct /= len(loader.dataset)
    it = iter(loader)
    img = next(it)[0]
    correct /= img.shape[-1] * img.shape[-2]
    acc = 100. * correct
    print('\n[{}] Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(subset,
        loss, correct, len(loader.dataset), acc))
    
    return loss, acc

# ------------------- CONSTANTS ------------------- #

MEAN=[94.7450, 102.9471, 96.9884]
STD=[31.5119, 24.6287, 19.7533]

LOG_INTERVAL = args.log_interval
TRAIN_TEST_RATIO = 0.8

CROP_SIZE = 128 # RandomCrop transform
BATCH_SIZE = 10
LR = args.lr
EPOCHS=args.epochs
SEED=args.seed

# ------------------- DIRECTORY PATHS ------------------- #

# ROOT_PATH = '/home/s2254242/PHD/ATI/peak-district-challenge'
ROOT_PATH = '/shared/miguel/peak-district-challenge'
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

M = image.shape[0]//16
N = image.shape[1]//16

tiles = [image[x:x+M,y:y+N] for x in range(0,image.shape[0],M) for y in range(0,image.shape[1],N)]
tiles2 = [image2[x:x+M,y:y+N] for x in range(0,image2.shape[0],M) for y in range(0,image2.shape[1],N)]
# tiles3 = [image3[x:x+M,y:y+N] for x in range(0,image3.shape[0],M) for y in range(0,image3.shape[1],N)]


length = len(ds)

# Do the train-test split
perm = torch.randperm(len(ds))

train_ds = torch.utils.data.Subset(ds, perm[:int(TRAIN_TEST_RATIO*length)])
val_ds = torch.utils.data.Subset(ds, perm[int(TRAIN_TEST_RATIO*length):])

# Define dataloaders
trainloader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, pin_memory=True,
                                          shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, pin_memory=True,
                                          shuffle=True, num_workers=2)


torch.manual_seed(SEED)

# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
device = torch.device("cpu")


# Train the model

# model = SiameseNetwork(output_dim=(1,CROP_SIZE*CROP_SIZE)).to(device)
# model = SNUNet_ECAM(3, 1).to(device)
model = UNet(n_channels=3, n_classes=1).to(device)
model.load_state_dict(torch.load('/shared/miguel/peak-district-challenge/results_inclass_mixing/model_40.pth'))
model.eval()

criterion = nn.MSELoss()#reduction='sum')

iterator = iter(valloader)
batch = next(iterator)
train_correct = 0
train_cum_loss = []

original_image, augmented_image, change_mask, mixing_img, fouriermask, class_change = combine_masks_prediction(batch, "continuous_fmix")
        
change_mask = change_mask.float()

original_image_transformed = extra_transforms(original_image)
augmented_image_transformed = extra_transforms(augmented_image)

# Send to CUDA device
original_image = original_image.to(device)
augmented_image = augmented_image.to(device)
change_mask = change_mask.to(device)
        
outputs = model(original_image.float(), augmented_image.float())
loss = criterion(outputs.squeeze(), change_mask)
        
pred = torch.where(outputs > 0.5, 1, 0)  # get the index of the max log-probability
change_mask = torch.where(change_mask > 0.5, 1, 0)
train_correct += pred.eq(change_mask.view_as(pred)).sum().item()

train_cum_loss.append(loss.sum().item())  # cumulative train loss

train_loss = sum(train_cum_loss) / len(trainloader.dataset)
train_correct /= len(trainloader.dataset)
train_correct /= CROP_SIZE*CROP_SIZE
train_acc = 100. * train_correct


print("Accuracy eval:", train_acc)


import matplotlib.pyplot as plt

i=0
        
# Plot the original image and mask
fig, ax = plt.subplots(3,3, figsize=(10,10))
ax[0][0].set_title('Original image 1')
ax[0][0].imshow(original_image[i].permute(1,2,0))#.type(torch.uint8))#, vmin=0, vmax=1)
ax[0][1].set_title('Mixing with image 2')
ax[0][1].imshow(mixing_img[i].permute(1,2,0))#.type(torch.uint8))#, vmin=0, vmax=1)
ax[0][2].set_title('Fourier generated mask')
ax[0][2].imshow(fouriermask[i], vmin=0, vmax=1)
# ax[0][1].imshow(batch[1][i])
# io.imshow(color.label2rgb(batch[1][i],batch[0][i].permute(1,2,0)))
# ax[0][1].set_title('Original mask 1')

ax[1][0].set_title('Augmentations image 1')
ax[1][0].imshow(original_image_transformed[i].permute(1,2,0))
ax[1][1].set_title('Augmentations image 2')
ax[1][1].imshow(augmented_image_transformed[i].permute(1,2,0))
ax[1][2].set_title('Class change')
ax[1][2].imshow(class_change[i])

ax[2][0].set_title('Change mask between 1 and 2 (GT)')
ax[2][0].imshow(change_mask[i], vmin=0, vmax=1)
ax[2][1].set_title('Predicted change')
ax[2][1].imshow(outputs[i].permute(1,2,0).detach().numpy())
ax[2][2].set_title('Predicted change between 0 and 1')
ax[2][2].imshow(outputs[i].permute(1,2,0).detach().numpy(), vmin=0, vmax=1)

print(outputs[i].max())
print(outputs[i].min())
print('----')
print(change_mask[i].max())
print(change_mask[i].min())
plt.show()