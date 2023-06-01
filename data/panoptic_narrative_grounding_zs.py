import os
import json
import torch
import os.path as osp
from skimage import io
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from data.get_fpn_data import fpn_data
from PIL import Image
import numpy as np


class PanopticNarrativeGroundingValDataset(Dataset):
    """Panoptic Narrative Grounding dataset."""

    def __init__(self, cfg, split, dataset, train=True):
        """
        Args:
            Args:
            cfg (CfgNode): configs.
            train (bool):
        """
        self.cfg = cfg
        self.train = train # True or False
        # split = 'val2017'
        self.split = split # train2017 or val2017
        self.dataset = dataset

        self.mask_transform = Resize((256, 256))

        self.annotation_path = osp.join('/media/disk1/public/refcoco/anns', '{0:s}.json'.format(self.dataset))

        self.panoptic_path = osp.join('/home/jjy/PPMN/datasets/coco/annotations', 'panoptic_segmentation', self.dataset)

        self.image_path = osp.join('/media/disk1/public/refcoco/images', 'train2014')


        self.annotations = self.load_json(self.annotation_path)[self.split]

        fpn_dataset, self.fpn_mapper = fpn_data(cfg, 'zs')
        self.fpn_dataset = {i['image_id']: i for i in fpn_dataset}

    ## General helper functions
    def load_json(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        return data

    def save_json(self, filename, data):
        with open(filename, "w") as f:
            json.dump(data, f)
    
    def resize_gt(self, img, interp, new_w, new_h):
        interp_method = interp if interp is not None else self.interp

        if img.dtype == np.uint8:
            if len(img.shape) > 2 and img.shape[2] == 1:
                pil_image = Image.fromarray(img[:, :, 0], mode="L")
            else:
                pil_image = Image.fromarray(img)
            pil_image = pil_image.resize((new_w, new_h), interp_method)
            ret = np.asarray(pil_image)
            if len(img.shape) > 2 and img.shape[2] == 1:
                ret = np.expand_dims(ret, -1)
        else:
            # PIL only supports uint8
            if any(x < 0 for x in img.strides):
                img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            shape = list(img.shape)
            shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
            img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
            _PIL_RESIZE_TO_INTERPOLATE_MODE = {
                Image.NEAREST: "nearest",
                Image.BILINEAR: "bilinear",
                Image.BICUBIC: "bicubic",
            }
            mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[interp_method]
            align_corners = None if mode == "nearest" else False
            img = F.interpolate(
                img, (self.new_h, self.new_w), mode=mode, align_corners=align_corners
            )
            shape[:2] = (self.new_h, self.new_w)
            ret = img.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)

        return ret

    def __len__(self):
        return len(self.annotations)
    
    def vis_item(self, img, gt, idx):
        save_dir = f'vis/{idx}'
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        import cv2
        cv2.imwrite(osp.join(save_dir,'img.png'), img.numpy().transpose(1, 2, 0))
        for i in range(len(gt)):
            if gt[i].sum() != 0:
                cv2.imwrite(osp.join(save_dir, f'gt_{i}.png'), gt[i].numpy()*255)
        
    def __getitem__(self, idx):

        annotation = self.annotations[idx]
        image_id = int(annotation['iid'])

        caption = annotation['refs'][0]

        segms = np.load(osp.join(self.panoptic_path, '{:s}.npy'.format(str(annotation['mask_id'])))).astype(np.float32)
        segms = torch.from_numpy(segms)[None, :, :]

        boxes = np.array(annotation['bbox'])

        h, w = segms.shape[1], segms.shape[2]

        boxes[0] = boxes[0] / w
        boxes[2] = boxes[2] / w
        boxes[1] = boxes[1] / h
        boxes[3] = boxes[3] / h

        fpn_data = self.fpn_mapper(self.fpn_dataset[image_id])  

        return caption, fpn_data, segms, boxes
