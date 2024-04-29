import wandb

from sam_lora.model import SamLora
from sam_lora.dataset import *
from torchvision import transforms
import argparse
from torch.utils.data import DataLoader
import pytorch_lightning as pl

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Test script that greets the user.")

    # args for model
    parser.add_argument("--vit_type", type=str, choices=["vit_t", "vit_b", "vit_l", "vit_h"],
                        help="Type of vit", required=True,default="vit_t")
    parser.add_argument("--lora_rank", type=int, help="rank of LoRA",required=False,default=2)
    parser.add_argument("--sam_checkpoint", type=str, help="path for sam checkpoint",required=True)

    # args for training
    parser.add_argument("--log_wandb", type=bool,help="log to wandb",required=False,default=False)
    parser.add_argument("--wandb_project", type=str,help="wandb project name",required=False,default="Sam_LoRA")
    parser.add_argument("--wandb_key", type=str,help="wandb user name key",required=False)
    parser.add_argument("--wandb_log_model", type=bool,help="wandb log model",required=False)


    parser.add_argument("--epochs", type=int, help="epochs_num",required=False,default=100)
    parser.add_argument("--T_max", type=int, help="T_max for schudler,if zero without",required=False,default=0)
    parser.add_argument("--lr", type=float, help="learning rate",required=False,default=0.001)


    parser.add_argument("--train_images_path", type=str, help="train images path",required=True)
    parser.add_argument("--train_masks_path", type=str, help="train masks path",required=True)
    parser.add_argument("--val_images_path", type=str, help="val images path",required=True)
    parser.add_argument("--val_masks_path", type=str, help="val masks path",required=True)
    parser.add_argument("--classes_labels_path", type=str, help="path for txt file of the classes",required=True)

    parser.add_argument("--images_and_mask_augmentation", type=bool, help="Do augmentation to mask and image(Rotation, rotate)",required=False,default=True)
    parser.add_argument("--p_flip", type=float, help="probibility to flip the mask and the image",required=False,default=0.5)
    parser.add_argument("--degrees_rotate", type=int, help="rotation to rotate",required=False,default=90)
    parser.add_argument("--image_augmentation", type=bool, help="do augmentation on the image",required=False,default=True)

    parser.add_argument("--device", type=str, help="device to train with",required=False,default="mps")

    args = parser.parse_args()

    image_only_transforms = None
    if args.image_augmentation:
        image_only_transforms = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.4)])

    dataset = SamDataset(args.train_images_path, args.train_masks_path,mask_and_image_transform=args.images_and_mask_augmentation,
                         p_flip = args.p_flip,degrees = args.degrees,
                         image_specific_transform=image_only_transforms)
    dataset_test = SamDataset(args.val_images_path, args.val_masks_path, mask_and_image_transform=False)


    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)
    model = SamLora(sam_type=args.vit_type, r=args.lora_rank, sam_checkpoint=args.sam_checkpoint,
                    log_wandb=args.log_wandb,T_max = args.T_max,
                    classes_labels_path=args.classes_labels_path,lr=args.lr)

    if args.log_wandb:
        wandb.login(key=args.wandb_key)
        run = wandb.init(project=args.wandb_project,config = {
            'vit_type':args.vit_type,
            'LoRA rank':args.lora_rank,
            'epochs':args.epochs,
            'T_max':args.T_max,

        })
        with run:
            logger = pl.loggers.WandbLogger(experiment=run, log_model=args.wandb_log_model)
            trainer = pl.Trainer(accelerator=args.device, max_epochs=args.epochs, logger=logger)
            trainer.fit(model, dataloader, dataloader_test)

    trainer = pl.Trainer(accelerator=args.device, max_epochs=args.epochs)
    trainer.fit(model, dataloader, dataloader_test)


if __name__ == "__main__":
    main()


