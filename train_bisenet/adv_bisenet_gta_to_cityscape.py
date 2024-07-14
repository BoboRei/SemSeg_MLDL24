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
    parser.add_argument('--lr_decay_rate', type=str, default=500, help='Lr decay rate')
    parser.add_argument('--save_d1_path', type=str, default=None, help='path to save d1')
    parser.add_argument('--save_d2_path', type=str, default=None, help='path to save d2')
    

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
    
    t_val = A.Compose([A.Resize(args.val_crop_height, args.val_crop_width, interpolation=cv2.INTER_NEAREST),])

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
     
    init_lr = args.init_lr
    num_steps = args.steps
    iter_size = args.iter_size
    lr1 = args.discriminator_lr
    lr2 = args.discriminator_lr
    lr_decay = args.lr_decay_rate
    checkpoint_step = args.checkpoint_step
    val_step = args.validation_step
    model_path_bisenet = args.save_model_path
    model_path_D1 = args.save_d1_path
    model_path_D2 = args.save_d2_path
    
    # Bisenet Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
     
    bisenet_model = bisenet.BiSeNet(19, 'resnet18').to(device)
    
    if args.Uploaded == 'True':
        bisenet_model.load_state(args.path_to_weights)
    
    model_D1 = discriminator.FCDiscriminator(num_classes=19).to(device)
    model_D2 = discriminator.FCDiscriminator(num_classes=19).to(device)
        
    # Define loss, optimizer
    optimizer = optim.AdamW(bisenet_model.parameters(), lr = args.init_lr, weight_decay = args.weight_decay)
    loss_fun = nn.CrossEntropyLoss(ignore_index = 255)
    num_epochs = args.num_epochs
 
    
    # Discriminator optimizer and loss
    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=lr1, betas=(0.9, 0.99))
    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=lr2, betas=(0.9, 0.99))
    bce_loss = torch.nn.BCEWithLogitsLoss()
    
    trainloader_iter = enumerate(train_dataloader)
    targetloader_iter = enumerate(val_dataloader)


    for i_iter in tqdm(range(num_steps)):
        train_bisenet(bisenet_model, optimizer, train_dataloader, loss_fun, device, i_iter, list_, init_lr, lr_1, lr_2, trainloader_iter, targetloader_iter, val_dataloader, iter_size, model_path_bisenet, model_path_D1, model_path_D2, num_steps, lr_decay, checkpoint_step)
        if (i_iter) % val_step == 0 or i_iter > num_steps-3:
            val_bisenet(model_resnet18, val_dataloader, loss_fun, device, list_)
    

    
def train_bisenet(model_resnet18, optimizer, train_dataloader, loss_fun, device, i_iter, list_, init_lr, lr_1, lr_2, trainloader_iter, targetloader_iter, val_dataloader, iter_size, model_path_bisenet, model_path_D1, model_path_D2, num_steps, lr_decay, checkpoint_step):
    model_resnet18.train()
    model_D1.train()
    model_D2.train()
    loss_seg_value1 = 0
    loss_adv_target_value1 = 0
    loss_D_value1 = 0
    loss_seg_value2 = 0
    loss_adv_target_value2 = 0
    loss_D_value2 = 0

    total_iou = 0
    total_batches = 0
    epoch_loss = 0
    iou = []
    
    optimizer.zero_grad()
    init_lr = poly_lr_scheduler(optimizer, init_lr, i_iter, lr_decay_iter=lr_decay, max_iter=num_steps, power=0.9)
    optimizer_D1.zero_grad()
    optimizer_D2.zero_grad()
    lr_1 = poly_lr_scheduler(optimizer_D1, lr_1, i_iter, lr_decay_iter=lr_decay, max_iter=num_steps, power=0.9)
    lr_2 = poly_lr_scheduler(optimizer_D2, lr_2, i_iter, lr_decay_iter=lr_decay, max_iter=num_steps, power=0.9)

    for sub_i in range(iter_size):
        for param in model_D1.parameters():
            param.requires_grad = False
        for param in model_D2.parameters():
            param.requires_grad = False

        try:
            _, batch = trainloader_iter.__next__()
        except StopIteration:
            trainloader_iter = enumerate(train_dataloader)
            _, batch = trainloader_iter.__next__()

        images, labels = batch
        images = images.to(device)
        labels = labels.long().to(device)

        try:
            batch_target = next(targetloader_iter)
            target_images = batch_target[1][0]
        except StopIteration:
            targetloader_iter = enumerate(val_dataloader)
            batch_target = next(targetloader_iter)
            target_images = batch_target[1][0]

        target_images = target_images.to(device)

        pred1, pred2 , _ = model_resnet18(images)
        pred1 = interp(pred1)
        
        loss_seg1 = loss_fun(pred1, labels)
        loss = loss_seg1 / iter_size
        loss.backward()
        loss_seg_value1 += loss_seg1.item() / iter_size
        
        # Compute training loss e mIoU
        epoch_loss += loss_seg1.item()
        predicted = pred1.detach().argmax(dim=1).cpu().numpy().astype(int)
        labels = labels.cpu().numpy().astype(int)
        
        hist = fast_hist(labels.squeeze(), predicted, 19)
        iou.append(per_class_iou(hist))
        batch_iou = np.mean(iou[total_batches])
        
        total_iou += batch_iou
        total_batches += 1
        
        # Train G with target
        try:
            _, batch = targetloader_iter.__next__()
        except StopIteration:
            targetloader_iter = enumerate(val_dataloader)
            _, batch = targetloader_iter.__next__()

        images, _ = batch
        images = images.to(device)

        pred_target1, pred_target2, _ = model_resnet18(images)
        pred_target1 = interp_target(pred_target1)
        pred_target2 = interp_target(pred_target2)

        D_out1 = model_D1(F.softmax(pred_target1))
        D_out2 = model_D2(F.softmax(pred_target2))

        loss_adv_target1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(device)) 
        loss_adv_target2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_label).to(device))
        
        loss = lambda_adv_target1 * loss_adv_target1 + lambda_adv_target2 * loss_adv_target2
        loss = loss / iter_size
        loss.backward()
        loss_adv_target_value1 += loss_adv_target1.item() / iter_size
        loss_adv_target_value2 += loss_adv_target2.item() / iter_size

        # Train discriminator D on source
        for param in model_D1.parameters():
            param.requires_grad = True
        for param in model_D2.parameters():
            param.requires_grad = True

        pred1 = pred1.detach()
        pred2 = pred2.detach()

        D_out1 = model_D1(F.softmax(pred1))
        D_out2 = model_D2(F.softmax(pred2))
        
        loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(device))
        loss_D2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_label).to(device))

        loss_D1 = loss_D1 / iter_size / 2
        loss_D2 = loss_D2 / iter_size / 2

        loss_D1.backward()
        loss_D2.backward()

        loss_D_value1 += loss_D1.item()
        loss_D_value2 += loss_D2.item()
        
        # Train discriminator D on train
        pred_target1 = pred_target1.detach()
        pred_target2 = pred_target2.detach()

        D_out1 = model_D1(F.softmax(pred_target1))
        D_out2 = model_D2(F.softmax(pred_target2))

        loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(target_label).to(device))
        loss_D2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(target_label).to(device))
        
        loss_D1 = loss_D1 / iter_size / 2
        loss_D2 = loss_D2 / iter_size / 2

        loss_D1.backward()
        loss_D2.backward()

        loss_D_value1 += loss_D1.item()
        loss_D_value2 += loss_D2.item()
        
    optimizer.step()
    optimizer_D1.step()
    optimizer_D2.step()

    if i_iter % checkpoint_step == 0 or  i_iter > num_steps-3:
        torch.save(model_resnet18.state_dict(), model_path_bisenet)
        torch.save(model_D1.state_dict(), model_path_D1)
        torch.save(model_D2.state_dict(), model_path_D2)
        print(f"Learning rate BiSenet at iteration {i_iter+1}: {init_lr}")
        print(f"Learning rate D1 at iteration {i_iter+1}: {lr_1}")
        print(f"Learning rate D2 at iteration {i_iter+1}: {lr_2}")

    # Total mIoU and training loss computation
    mIOU = (total_iou / total_batches) * 100
    training_loss = epoch_loss / len(train_dataloader)

    print(f"Iteration: {i_iter+1}, Training Loss: {training_loss:.4f}, mIoU: {mIOU:.2f}%")
    if (i_iter + 1) % checkpoint_step == 0 or i_iter > num_steps-3:
        for i in range(len(list_)):
            print(f"Accuracy {list_[i]}: IOU: {100 * np.mean([vector[i] for vector in iou]):.2f}%")      
            

    def val_bisenet(bisenet_model, val_dataloader, loss_fun, device):
        bisenet_model.eval()
        val_loss = 0.0
        total_iou = 0
        total_batches = 0

        with torch.no_grad():
            for batch, (image, mask) in enumerate(val_dataloader):
                image, mask = image.to(device), mask.to(device)
                mask = mask.type(torch.long)
                mask_pred = bisenet_model(image)
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

    

if __name__ == '__main__':
    main()