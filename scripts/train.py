import land_cover_models as lcm
import torch
import torch.nn as nn
from torchvision import transforms, utils
from fmix import combine_masks
from my_transformations import RandomCrop, extra_transforms
from models import SiameseNetwork
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import matplotlib.pyplot as plt
import argparse
from unet import UNet
import wandb

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
BATCH_SIZE = args.batch_size
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

wandb.init(project='peak-district', entity="mespinosami")
wandb.config.update(args)


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

length = len(ds)

# Do the train-test split
perm = torch.randperm(len(ds))

train_ds = torch.utils.data.Subset(ds, perm[:int(TRAIN_TEST_RATIO*length)])
val_ds = torch.utils.data.Subset(ds, perm[int(TRAIN_TEST_RATIO*length):])

# Define dataloaders
trainloader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, pin_memory=True,
                                          shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, pin_memory=True,
                                          shuffle=True, num_workers=2)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



# Train the model

# model = SiameseNetwork(output_dim=(1,CROP_SIZE*CROP_SIZE)).to(device)
# model = SNUNet_ECAM(3, 1).to(device)
model = UNet(n_channels=3, n_classes=1).to(device)

if args.optim == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
elif args.optim == 'adamw':
    optimizer = optim.AdamW(model.parameters(), lr=LR)
else:
    optimizer = optim.Adadelta(model.parameters(), lr=LR)

# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
# scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS*(len(trainloader)//BATCH_SIZE), eta_min=0.00001)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS*len(trainloader), eta_min=0.00001)
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.MSELoss()#reduction='sum')

for epoch in range(1, EPOCHS + 1):
    
    train_cum_loss = []
    train_correct = 0

    # Training loop
    for batch_idx, batch in enumerate(trainloader):
        # Create the artificial augmented image and get the change mask
        original_image, augmented_image, change_mask = combine_masks(batch, "continuous_fmix")
        
        change_mask = change_mask.float()
        
        original_image_transformed = extra_transforms(original_image)
        augmented_image_transformed = extra_transforms(augmented_image)
        
        # Send to CUDA device
        original_image = original_image.to(device)
        augmented_image = augmented_image.to(device)
        change_mask = change_mask.to(device)
        
        optimizer.zero_grad()
        outputs = model(original_image.float(), augmented_image.float()).squeeze()
        loss = criterion(outputs, change_mask)
        # print(outputs.shape)
        # print(change_mask.shape)
        
        pred = torch.where(outputs > 0.5, 1, 0)  # get the index of the max log-probability
        change_mask = torch.where(change_mask > 0.5, 1, 0)
        train_correct += pred.eq(change_mask.view_as(pred)).sum().item()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        wandb.log({'lr': scheduler.get_last_lr()[0]})
        
        train_cum_loss.append(loss.sum().item())  # cumulative train loss
        
        
        if batch_idx % LOG_INTERVAL == 0:
            # wandb.log({'batch': epoch*(len(trainloader)//(BATCH_SIZE+LOG_INTERVAL))+batch_idx,
            #         'train_cum_loss': train_cum_loss[-1]/LOG_INTERVAL})
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(original_image), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item()))
    
    
    train_loss = sum(train_cum_loss) / len(trainloader.dataset)
    train_correct /= len(trainloader.dataset)
    train_correct /= CROP_SIZE*CROP_SIZE
    train_acc = 100. * train_correct
    
    
    model.eval()
    with torch.no_grad():
        val_loss, val_acc = compute_loss_and_acc(valloader, model, criterion, subset='validation')
    
    try:
        wandb.log({"epoch": epoch, # "lr": scheduler.get_last_lr()[0],
                   "train_acc": train_acc, "train_loss": train_loss,
                   "val_acc": val_acc, "val_loss": val_loss})
    except ValueError:
        print(f"Invalid stats?")
    
    
    if epoch % 20 == 0:
        torch.save(model.state_dict(), f"{args.save_model_path}/model_{epoch}.pth")