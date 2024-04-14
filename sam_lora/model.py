import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from peft import get_peft_model,LoraConfig
from mobile_sam import sam_model_registry
from sam_lora.losses import *
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
from sam_lora.dataset import *
from torch.utils.data import DataLoader
import wandb
import torchvision.transforms as transforms

class SamLora(pl.LightningModule):
    def __init__(self, r=2, sam_type='vit_t', sam_checkpoint='', log_wandb=True, T_max=20, classes_labels_path = ""):
        super(SamLora, self).__init__()
        self.log_wandb = log_wandb
        self.model = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
        self.r = r
        self.T_max = T_max
        all_linear = ['qkv', 'proj', 'fc1', 'fc2', 'head', 'q_proj', 'k_proj', 'v_proj', 'out_proj', 'lin1', 'lin2']
        target_modules = all_linear
        peft_config = LoraConfig(
            r=self.r,
            lora_alpha=self.r,
            bias="none",
            target_modules=target_modules,
            init_lora_weights='pissa',
        )
        self.model = get_peft_model(self.model, peft_config)
        self.labels_name = []
        self.labels_id = []
        with open(classes_labels_path, 'r') as file:
            for line in file:
                # Split the line into parts
                self.labels_id.append(int(line.split('-')[0]))
                self.labels_name.append(line.split('-')[1])

        self.input_points,self.input_label = torch.randint(0, 1000, (len(self.labels_id), 1, 2)),torch.ones((len(self.labels_id),1))

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def map_gt_masks_to_model_labels(self,gt_masks, gt_labels, model_labels):
        output_masks = []
        for i,value in enumerate(model_labels):
            if value in gt_labels:
                output_masks.append(gt_masks[gt_labels.index(value)])
            else:
                output_masks.append(torch.zeros_like(gt_masks[0]))
        output_masks = torch.stack(output_masks).squeeze(1)
        return output_masks


    def training_step(self, batch, batch_idx):
        # Training step
        x = batch
        input_points = self.input_points.clone().to(x['image'].device)
        gt_masks = self.map_gt_masks_to_model_labels(x['mask'][0],x['unique_ids'][0].tolist(),[79,96,170,213])

        input_labels = torch.tensor(self.input_label).to(x['image'].device)
        x_input = [{'image': x['image'][0], 'point_coords': input_points, 'point_labels': input_labels,
                    'original_size': (x['image'].shape[-2], x['image'].shape[-1])}]

        sam_output = self.model(x_input, multimask_output=False)[0]

        gt_masks = gt_masks.view(sam_output['masks'].shape).to(x['image'].device)
        dice_loss, _ = cal_dice_loss(gt_masks, x['mask'])
        f_loss = focal_loss(sam_output['masks_grad'], gt_masks)
        _, real_iou = soft_iou_loss(sam_output['masks_grad'], gt_masks)

        iou_predection_loss = nn.MSELoss()(sam_output['iou_predictions'], real_iou)

        total_loss = dice_loss + 20 * f_loss

        if self.log_wandb:
            if batch_idx % 10 == 0:
                sam_masks_copy = sam_output['masks'].clone().detach().cpu().bool()
                image_tensor = x['image'][0].clone()
                gt_masks_copy = gt_masks.clone().detach().cpu().bool()
                sam_masks_image = overlay_masks(image_tensor, sam_masks_copy)
                gt_masks_image = overlay_masks(image_tensor, gt_masks_copy)
                self.logger.log_image(key="ground_truth ,predicted", images=[gt_masks_image, sam_masks_image])
            if batch_idx % 2 == 0:
                self.logger.log_metrics(
                    {'dice_loss': dice_loss, 'total_loss': total_loss, 'iou_predection_loss': iou_predection_loss,
                     'focal_loss': f_loss})

        self.log('total_loss', total_loss, prog_bar=True)
        self.training_step_outputs.append(total_loss)
        return total_loss

    def on_train_epoch_end(self):
        all_preds = torch.stack(self.training_step_outputs)
        avg_loss = torch.mean(all_preds)
        self.logger.log_metrics({'avg_loss_train_epoch': avg_loss})
        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        # Validation step
        x = batch
        input_points = self.input_points.clone().to(x['image'].device)
        input_labels = torch.tensor(self.input_label).to(x['image'].device)

        x_input = [{'image': x['image'][0], 'point_coords': input_points, 'point_labels': input_labels,
                    'original_size': (x['image'].shape[-2], x['image'].shape[-1])}]

        sam_output = self.model(x_input, multimask_output=False)[0]

        gt_masks = self.map_gt_masks_to_model_labels(x['mask'][0], x['unique_ids'][0].tolist(), self.labels_id)
        gt_masks = gt_masks.view(sam_output['masks'].shape).to(x['image'].device)

        dice_loss, _ = cal_dice_loss(gt_masks, x['mask'])
        f_loss = focal_loss(sam_output['masks_grad'], gt_masks)
        _, real_iou = soft_iou_loss(sam_output['masks_grad'], gt_masks)

        iou_predection_loss = nn.MSELoss()(sam_output['iou_predictions'], real_iou)

        total_loss = dice_loss + 20 * f_loss

        # Optionally, you can calculate additional metrics like IoU here as well
        if self.log_wandb:
            t = 0
            if self.current_epoch % 2 == 0:
                t = 1
            if batch_idx % 4 == t:
                sam_masks_copy = sam_output['masks'].clone().detach().cpu().bool()
                image_tensor = x['image'][0].clone()
                gt_masks_copy = gt_masks.clone().detach().cpu().bool()
                sam_masks_image = overlay_masks(image_tensor, sam_masks_copy)
                gt_masks_image = overlay_masks(image_tensor, gt_masks_copy)
                self.logger.log_image(key="ground_truth_real ,predicted_real", images=[gt_masks_image, sam_masks_image])
        self.validation_step_outputs.append(total_loss)
        return total_loss

    def on_validation_epoch_end(self):
        all_preds = torch.stack(self.validation_step_outputs)
        avg_loss = torch.mean(all_preds)
        self.log('avg_loss_val_epoch', avg_loss, prog_bar=True)
        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):

        # Use AdamW optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)

        # Add Cosine Annealing Learning Rate Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=self.T_max)  # T_max is the number of steps until the next restart

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def configure_callbacks(self):
        # Callback to save the model based on minimum val_loss
        checkpoint_callback = ModelCheckpoint(
            monitor='avg_loss_val_epoch',  # Monitoring validation loss
            filename='checkpoint-{epoch:02d}-{avg_loss_val_epoch:.2f}',
            save_top_k=1,  # Number of best models to save; set to 1 to save the best model only
            mode='min',  # Mode 'min' will save the model when val_loss has decreased
            save_last=True  # Save the last model in addition to the best model
        )
        # Callback for early stopping
        early_stopping_callback = EarlyStopping(
            monitor='avg_loss_val_epoch',  # Monitor validation loss
            patience=120,  # Number of epochs with no improvement after which training will be stopped
            mode='min'  # Mode 'min' to monitor decreasing validation loss
        )

        return [checkpoint_callback, early_stopping_callback]



#if __name__ == 'main':
images_dir = '/Users/mac/Documents/datasets/football_segmentation/images'
masks_dir = '/Users/mac/Documents/datasets/football_segmentation/masks'

images_test_dir = '/Users/mac/Documents/datasets/football_segmentation/images'
masks_test_dir = '/Users/mac/Documents/datasets/football_segmentation/masks'

classes_labels_path = "/Users/mac/Documents/datasets/football_segmentation/labels"

model_path = '/Users/mac/PycharmProjects/ALL_Shit/checkpoints/SAM/mobile_sam.pt'

image_only_transforms = transforms.Compose([
    transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)], p=0.5),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.4)])

dataset = SamDataset(images_dir, masks_dir, mask_and_image_transform=True,image_specific_transform= image_only_transforms)
dataset_test = SamDataset(images_test_dir, masks_test_dir, mask_and_image_transform=False)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)


if 1:
    model = SamLora(sam_checkpoint=model_path, log_wandb=True, classes_labels_path=classes_labels_path)

    run = wandb.init(project='football_segmentation')
    with run:
        logger = pl.loggers.WandbLogger(experiment=run, log_model=False)
        trainer = pl.Trainer(accelerator='mps', max_epochs=200,logger=logger)
        trainer.fit(model,dataloader,dataloader)
else:

    model = SamLora(sam_checkpoint=model_path, log_wandb=False, classes_labels_path=classes_labels_path)
    trainer = pl.Trainer(accelerator='mps', max_epochs=200)
    trainer.fit(model,dataloader,dataloader_test)



