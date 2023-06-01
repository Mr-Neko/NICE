
## Installation

### Requirements

- Python
- Numpy
- Pytorch 1.7.1
- Tqdm 4.56.0
- Scipy 1.5.3
- scikit-image
- scikit-learn

### Cloning the repository

```
$ git clone git@github.com:Mr-Neko/png.git
$ cd png
```

## Dataset Preparation

### Panoptic Narrative Grounding Benchmark

1. Download the 2017 MSCOCO Dataset from its [official webpage](https://cocodataset.org/#download). You will need the train and validation splits' images and panoptic segmentations annotations.

2. Download the Panoptic Narrative Grounding Benchmark from the PNG's [project webpage](https://bcv-uniandes.github.io/panoptic-narrative-grounding/#downloads). Organize the files as follows:

```
panoptic_narrative_grounding
|_ images
|  |_ train2017
|  |_ val2017
|_ annotations
   |_ png_coco_train2017.json
   |_ png_coco_val2017.json
   |_ panoptic_segmentation
   |  |_ train2017
   |  |_ val2017
   |_ panoptic_train2017.json
   |_ panoptic_val2017.json
```

3. Pre-process the Panoptic narrative Grounding Ground-Truth Annotation for the dataloader using [utils/pre_process.py](utils/pre_process.py).

4. At the end of this step you should have two new files in your annotations folder.

```
panoptic_narrative_grounding
|_ annotations
   |_ png_coco_train2017.json
   |_ png_coco_val2017.json
   |_ png_coco_train2017_dataloader.json
   |_ png_coco_val2017_dataloader.json
   |_ panoptic_segmentation
   |  |_ train2017
   |  |_ val2017
   |_ panoptic_train2017.json
   |_ panoptic_val2017.json
```

## Train and Inference

Modify the routes in [train_net.sh](train_net.sh) according to your local paths. If you want to only test the pretrained model, add `--ckpt_path ${PRETRAINED_MODEL_PATH}` and `--test_only`.

## Pretrained Bert Model
The bert can be downloaded from HuggingFace, and fpn model should be downloaded from https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl
```
pretrained_models
|_fpn
|  |_model_final_cafdb1.pkl
|_bert
|  |_bert-base-uncased
|  |  |_pytorch_model.bin
|  |  |_bert_config.json
|  |_bert-base-uncased.txt
```

## Acknowledge

Some of the codes are built upon [K-Net](https://github.com/ZwwWayne/K-Net) and [PNG](https://github.com/BCV-Uniandes/PNG). Thanks them for their great works!
