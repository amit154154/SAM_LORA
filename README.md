# Few-Shot Segmentation with SAM and LoRA

## Overview

The project leverages the segmentation capabilities of SAM, allowing for the segmentation of images based on different class prompts. Each segmentation class receives a different prompt encoding (from a random positive point), and the model is fine-tuned using LoRA to produce the desired segmentation mask based on this encoding.

## Key Ideas

Using LoRA we can leverage the SAM model to segment classes we want using very few images with no
use of any prompt(given manual or using a detection model such as grounding sam, yolo world etc...)
previous works in the subject such as PerSAM gives worst results and cant utilize diffrent classes

## Example

I trained using this method on few shot  [football dataset](https://www.kaggle.com/datasets/sadhliroomyprime/football-semantic-segmentation)

I trained using vit_t (MobileSAM) with LoRA rank 8, with those classes:

1. Advertisement
2. Goal Bar
3. ball
4. ground
5. team red
6. team black
7. refree

### more detils

I use the same loss as in the paper of SAM (except mse between the predicted iou to), for now it
generate random point to encode for each class.

## TBD


- [ ] test on coco dataset with diffrent amount of annotations
- [ ] add diffrent loss, for class precision 
- [ ] implament for efficentvit SAM
- [ ] test diffrent LoRA ranks
- [ ] test PISSA (which is what right now) vs regular LoRA


## Getting Started

### Installation

Clone the repository
```bash
git clone https://github.com/amit154154/SAM_LORA.git
```

### set the dataset

sort your dataset such that there is a folder of train images, folder of segmentation masks and the same for training.
create a txt file such that each line is the class id (which it has in the segmentation masks) and the class name.
look at the example.


