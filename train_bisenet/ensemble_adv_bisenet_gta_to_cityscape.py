import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchmetrics
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import sys
import os
import glob
from datasets import cityscape_dataset, gta_dataset
from models import bisenet
from utils import poly_lr_scheduler, fast_hist, per_class_iou
from tqdm import tqdm
import wandb
import discriminator
import fda


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=5217, help='Number of steps to train for')
    parser.add_argument('--iter_size', type=int, default=3, help='Number of batch slide')
    parser.add_argument('--checkpoint_step', type=int, default=500, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=500, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="Cityscapes", help='Dataset you are using.')
    parser.add_argument('--train_crop_height', type=int, default=720, help='Train Height of cropped/resized input image to network')
    parser.add_argument('--train_crop_width', type=int, default=1280, help='Train Width of cropped/resized input image to network')
    parser.add_argument('--val_crop_height', type=int, default=512, help='Val Height of cropped/resized input image to network')
    parser.add_argument('--val_crop_width', type=int, default=1024, help='Val Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of images in each batch')
    parser.add_argument('--init_lr', type=float, default=0.001, help='learning rate used for train')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay used for train')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--image_train_path', type=str, default='GTA5/images', help='images training path')
    parser.add_argument('--mask_train_path', type=str, default='GTA5/labels', help='masks training path')
    parser.add_argument('--image_val_path', type=str, default='Cityspaces/images/val', help='images validation path')
    parser.add_argument('--mask_val_path', type=str, default='Cityspaces/gtFine/val', help='mask validation path')
    parser.add_argument('--wandb_key', type=str, default='', help='wandb key')
    parser.add_argument('--augmentation', type=str, default='', help='Which augmentation to put')
    parser.add_argument('--Uploaded', type=str, default='False', help='To upload pre-trained weights')
    parser.add_argument('--path_to_weights', type=str, default='', help='Path to pre-trained weights')
    parser.add_argument('--discriminator_lr', type=str, default=0.0001, help='Discriminator lr')
    parser.add_argument('--lambda_adv1', type=str, default=0.0002, help='Weight of discrimanator1 loss')
    parser.add_argument('--lambda_adv2', type=str, default=0.001, help='Weight of discrimanator2 loss')
    parser.add_argument('--lr_decay_rate', type=int, default=500, help='Lr decay rate')
    parser.add_argument('--save_d1_path', type=str, default=None, help='path to save d1')
    parser.add_argument('--save_d2_path', type=str, default=None, help='path to save d2')
    parser.add_argument('--paths_to_models', type=str, default='', help='Paths to the weights on various networks')
    
    return parser.parse_args()


args = parse_args()

def main():
    augmentation = args.augmentation
    if augmentation == '':
        t_train = A.Compose([A.Resize(args.train_crop_height, args.train_crop_width, interpolation=cv2.INTER_NEAREST),])
    elif augmentation =='Horizontal':
        t_train = A.Compose([
                        A.Resize(args.train_crop_height, args.train_crop_width, interpolation=cv2.INTER_NEAREST),
                        A.HorizontalFlip(p=0.5),
            ], additional_targets={'mask': 'mask'})

    elif augmentation == 'HSV':
        t_train = A.Compose([
                        A.Resize(args.train_crop_height, args.train_crop_width, interpolation=cv2.INTER_NEAREST),
            A.OneOrOther(
                A.Compose([
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                         ]),
                A.NoOp(),
    )
            ], additional_targets={'mask': 'mask'})

    elif augmentation == 'Both':
        t_train = A.Compose([
                        A.Resize(args.train_crop_height, args.train_crop_width, interpolation=cv2.INTER_NEAREST),
                        A.HorizontalFlip(p=0.5),
            A.OneOrOther(
                A.Compose([
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                         ]),
                A.NoOp(),
    )
            ], additional_targets={'mask': 'mask'})
    
    t_val = {
    'image': A.Compose([A.Resize(512,1024, interpolation=cv2.INTER_NEAREST), 
                       ]),
    'mask': A.Compose([A.Resize(512,1024, interpolation=cv2.INTER_NEAREST), 
                      ])
    }
    
    # Directiories
    image_train_path = args.image_train_path
    mask_train_path = args.mask_train_path
    image_val_path = args.image_val_path
    mask_val_path = args.mask_val_path

    train_dataset = gta_dataset.GTA(image_train_path, mask_train_path, t_train)
    val_dataset = cityscape_dataset.CityScapes(image_val_path, mask_val_path, t_val)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    list_ = [
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "light",
        "sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motocycle",
        "bicycle"
    ]

    model_paths = [args.paths_to_models]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_model(path, device) for path in model_paths]
    loss_fun = nn.CrossEntropyLoss(ignore_index = 255)

    # Ensembled validation
    avg_val_loss, mIOU = val_ensemble(models, val_dataloader, loss_fun, device)
    


# BiSeNet model load
def load_model(weight_path, device):
    weight_path = args.path_to_weights
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BiSeNet(19, 'resnet18').to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Function to get predictions from a single image
def get_predictions(model, image, device):
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
    return output



def val_ensemble(models, val_dataloader, loss_fun, device):
    for model in models:
        model.eval()
    iou_list = []  # Modificato da `iou` a `iou_list`
    val_loss = 0.0
    total_iou = 0
    total_batches = 0

    with torch.no_grad():
        for batch, (images, masks) in enumerate(tqdm(val_dataloader)):
            images, masks = images.to(device), masks.to(device)
            masks = masks.type(torch.long)
            
            # Ottieni le predizioni da tutti i modelli
            predictions = [get_predictions(model, images, device) for model in models]
            
            # Calcola la media delle predizioni
            mean_predictions = torch.mean(torch.stack(predictions), dim=0)
            
            # Calcola la perdita
            loss = loss_fun(mean_predictions, masks.squeeze())
            val_loss += loss.item()

            # Ottieni le predizioni finali
            predicted = mean_predictions.detach().argmax(dim=1)
            predicted = predicted.detach().cpu().numpy().astype(int)
            masks = masks.detach().cpu().numpy().astype(int)

            # Calcola l'IoU
            hist = fast_hist(masks.squeeze(), predicted, 19) 
            iou = per_class_iou(hist)
            iou_list.append(iou)  # Aggiungi l'IoU alla lista
            batch_iou = np.mean(iou)

            total_iou += batch_iou
            total_batches += 1

    avg_val_loss = val_loss / len(val_dataloader)
    mIOU = (total_iou / total_batches) * 100
    
    print(f'Validation Loss: {avg_val_loss:.6f}, mIOU: {mIOU:.2f}%')
    for i in range(len(list_)):
        print(f"Accuracy {list_[i]}: IOU: {100*np.mean([vettore[i] for vettore in iou_list]):.2f}%"  )  # Modificato `iou` a `iou_list`
    return avg_val_loss, mIOU