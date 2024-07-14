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
from datasets import cityscape_dataset
from models import deeplabv2
from utils import poly_lr_scheduler, fast_hist, per_class_iou
from tqdm import tqdm
import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--checkpoint_step', type=int, default=10, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=10, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="Cityscapes", help='Dataset you are using.')
    parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=1024, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of images in each batch')
    parser.add_argument('--init_lr', type=float, default=0.001, help='learning rate used for train')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay used for train')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--image_train_path', type=str, default='Cityspaces/images/train', help='images training path')
    parser.add_argument('--mask_train_path', type=str, default='Cityspaces/gtFine/train', help='masks training path')
    parser.add_argument('--image_val_path', type=str, default='Cityspaces/images/val', help='images validation path')
    parser.add_argument('--mask_val_path', type=str, default='Cityspaces/gtFine/val', help='mask validation path')
    parser.add_argument('--wandb_key', type=str, default='', help='wandb key')

    return parser.parse_args()

args = parse_args()


def main():
    t_train = A.Compose([A.Resize(args.crop_height, args.crop_width, interpolation=cv2.INTER_NEAREST),])
    t_val = A.Compose([A.Resize(args.crop_height, args.crop_width, interpolation=cv2.INTER_NEAREST),])

    # Directiories
    image_train_path = args.image_train_path
    mask_train_path = args.mask_train_path
    image_val_path = args.image_val_path
    mask_val_path = args.mask_val_path

    train_dataset = cityscape_dataset.CityScapes(image_train_path, mask_train_path, t_train)
    val_dataset = cityscape_dataset.CityScapes(image_val_path, mask_val_path, t_val)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)


    # DeepLabV2 Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    deeplab_model = deeplabv2.get_deeplab_v2(pretrain_model_path=args.pretrained_model_path).to(device)

    # Define loss, optimizer
    optimizer = optim.AdamW(deeplab_model.parameters(), lr = args.init_lr, weight_decay = args.weight_decay)
    loss_fun = nn.CrossEntropyLoss(ignore_index = 255)
    num_epochs = args.num_epochs
    
    # Training

    wandb.init(project="deeplab-segmentation")
    wandb.login(key=args.wandb_key)

    def train_deeplab(deeplab_model, optimizer, train_dataloader, loss_fun, device, epoch, num_epochs):
        training_loss = 0
        deeplab_model.train() 
        total_iou = 0
        total_batches = 0

        for batch, (image, mask) in enumerate(train_dataloader):
            image, mask = image.to(device), mask.to(device)
            mask = mask.type(torch.long)

            optimizer.zero_grad()
            mask_pred, _, _ = deeplab_model(image)
            loss = loss_fun(mask_pred, mask.squeeze())
            loss.backward()
            optimizer.step()

            
            lr = poly_lr_scheduler(optimizer, init_lr, epoch, epochs_decay, num_epochs)

            predicted = mask_pred.detach().argmax(dim=1)
            predicted = predicted.detach().cpu().numpy().astype(int)
            mask = mask.detach().cpu().numpy().astype(int)
            
            hist = fast_hist(mask.squeeze(), predicted, 19)
            iou = per_class_iou(hist)
            batch_iou = np.mean(iou)

            total_iou += batch_iou
            total_batches += 1

            training_loss += loss.item()

        mIOU = (total_iou / total_batches) * 100
        avg_training_loss = training_loss / len(train_dataloader)

        print(f"Epoch: {epoch+1}, Training Loss: {avg_training_loss}, mIOU: {mIOU:.2f}% \n")
        if epoch % args.checkpoint_step == 0 or epoch > num_epochs-2:
            torch.save(deeplab_model.state_dict(), args.save_model_path)

        
        wandb.log({"Training Loss": avg_training_loss, "Training mIOU": mIOU})

        return image, mask, predicted


    def val_deeplab(deeplab_model, val_dataloader, loss_fun, device):
        deeplab_model.eval()
        val_loss = 0.0
        total_iou = 0
        total_batches = 0

        with torch.no_grad():
            for batch, (image, mask) in enumerate(val_dataloader):
                image, mask = image.to(device), mask.to(device)
                mask = mask.type(torch.long)
                mask_pred = deeplab_model(image)
                loss = loss_fun(mask_pred, mask.squeeze())
                val_loss += loss.item()
                predicted = mask_pred.detach().argmax(dim=1)
                predicted = predicted.detach().cpu().numpy().astype(int)
                mask = mask.detach().cpu().numpy().astype(int)
                hist = fast_hist(mask.squeeze(), predicted, 19)
                iou = per_class_iou(hist)
                batch_iou = np.mean(iou)

                total_iou += batch_iou
                total_batches += 1

        avg_val_loss = val_loss / len(val_dataloader)
        mIOU = (total_iou / total_batches) * 100
        
        print(f'Validation Loss: {avg_val_loss:.6f}, mIOU: {mIOU:.2f}%')

        
        wandb.log({"Validation Loss": avg_val_loss, "Validation mIOU": mIOU})

    for epoch in tqdm(range(num_epochs)):
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        image, mask, mask_pred = train_deeplab(deeplab_model, optimizer, train_dataloader, loss_fun, device, epoch, num_epochs)
         if epoch % args.validation_step == 0 or epoch > num_epoch-2:
            val_deeplab(deeplab_model, val_dataloader, loss_fun, device)
    
    wandb.finish()



if __name__ == '__main__':
    main()