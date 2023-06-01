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
from torchvision import transforms
from models.encoder_bert import convert_sents_to_features
from models.tokenization import BertTokenizer
import xml.etree.ElementTree as ET
import random

class PanopticNarrativeGroundingDataset(Dataset):
    """Panoptic Narrative Grounding dataset."""

    def __init__(self, cfg, split, train=True):
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

        self.mask_transform = Resize((256, 256))

        self.ann_dir = osp.join(cfg.data_path, "annotations")
        self.panoptic = self.load_json(
            osp.join(self.ann_dir, "panoptic_{:s}.json".format(split))
        )
        self.images = self.panoptic["images"]
        self.images = {i["id"]: i for i in self.images}
        self.panoptic_anns = self.panoptic["annotations"]
        self.panoptic_anns = {a["image_id"]: a for a in self.panoptic_anns}
        if not osp.exists(
            osp.join(self.ann_dir, 
                "png_coco_{:s}_dataloader.json".format(split),)
        ):
            print("No such a dataset")
        else:
            self.panoptic_narrative_grounding = self.load_json(
                osp.join(self.ann_dir, 
                    "png_coco_{:s}_dataloader.json".format(split),)
            )
        self.panoptic_narrative_grounding = [
            ln
            for ln in self.panoptic_narrative_grounding
            if (
                torch.tensor(np.array([item for sublist in ln["labels"] 
                    for item in sublist]))
                != -2
            ).any()
        ]
        fpn_dataset, self.fpn_mapper = fpn_data(cfg, split[:-4])
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
        return len(self.panoptic_narrative_grounding)
    
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
        localized_narrative = self.panoptic_narrative_grounding[idx]
        caption = localized_narrative['caption']
        image_id = int(localized_narrative['image_id'])
        fpn_data = self.fpn_mapper(self.fpn_dataset[image_id])  
        image_info = self.images[image_id]
        labels = localized_narrative['labels']

        noun_vector = localized_narrative['noun_vector']
        if len(noun_vector) > (self.cfg.max_sequence_length - 2):
            noun_vector_padding = \
                    noun_vector[:(self.cfg.max_sequence_length - 2)]
        elif len(noun_vector) < (self.cfg.max_sequence_length - 2): 
            noun_vector_padding = \
                noun_vector + [0] * (self.cfg.max_sequence_length - \
                    2 - len(noun_vector))
        noun_vector_padding = [0] + noun_vector_padding + [0]
        noun_vector_padding = torch.tensor(np.array(noun_vector_padding)).long()
        assert len(noun_vector_padding) == \
            self.cfg.max_sequence_length
        
        ret_noun_vector = noun_vector_padding[noun_vector_padding.nonzero()].flatten()
        assert len(ret_noun_vector) <= self.cfg.max_seg_num
        if len(ret_noun_vector) < self.cfg.max_seg_num:
            ret_noun_vector = torch.cat([ret_noun_vector, \
                ret_noun_vector.new_zeros((self.cfg.max_seg_num - len(ret_noun_vector)))])
        # ret_noun_vector: [max_seg_num]

        ann_types = [0] * len(labels)
        for i, l in enumerate(labels):
            l = torch.tensor(np.array(l))
            if (l != -2).any():
                ann_types[i] = 1 if (l != -2).sum() == 1 else 2
        ann_types = torch.tensor(np.array(ann_types)).long()
        ann_types = ann_types[ann_types.nonzero()].flatten()
        assert len(ann_types) <= self.cfg.max_seg_num
        if len(ann_types) < self.cfg.max_seg_num:
            ann_types = torch.cat([ann_types, \
                ann_types.new_zeros((self.cfg.max_seg_num - len(ann_types)))])
        
        ann_categories = torch.zeros([
            self.cfg.max_seg_num]).long()
        panoptic_ann = self.panoptic_anns[image_id]
        panoptic_segm = io.imread(
            osp.join(
                self.ann_dir,
                "panoptic_segmentation",
                self.split,
                "{:012d}.png".format(image_id),
            )
        )
        panoptic_segm = (
            panoptic_segm[:, :, 0]
            + panoptic_segm[:, :, 1] * 256
            + panoptic_segm[:, :, 2] * 256 ** 2
        )
        grounding_instances = torch.zeros(
            [self.cfg.max_seg_num, image_info['height'], image_info['width']]
        )
        grounding_boxes = torch.zeros(
            [self.cfg.max_seg_num, 4]
        )
        j = 0
        for i, bbox in enumerate(localized_narrative["boxes"]):
            box = torch.zeros(4)
            box[0] = image_info['width']
            box[1] = image_info['height']
            for b in bbox:
                if b != [0] * 4:
                    segment_info = [
                        s for s in panoptic_ann["segments_info"] 
                        if s["bbox"] == b
                    ][0]
                    segment_cat = [
                        c
                        for c in self.panoptic["categories"]
                        if c["id"] == segment_info["category_id"]
                    ][0]

                    if box[0] > b[0]:
                        box[0] = b[0]
                    if box[1] > b[1]:
                        box[1] = b[1]
                    if box[2] < b[0] + b[2]:
                        box[2] = b[0] + b[2]
                    if box[3] < b[1] + b[3]:
                        box[3] = b[1] + b[3]
                    instance = torch.zeros([image_info['height'],
                            image_info['width']])
                    instance[panoptic_segm == segment_info["id"]] = 1
                    grounding_instances[j, :] += instance
                    ann_categories[j] = 1 if \
                            segment_cat["isthing"] else 2
            if grounding_instances[j].sum() != 0:
                box[2] = (box[2]-box[0]) / image_info['width']
                box[3] = (box[3]-box[1]) / image_info['height']
                box[0] = box[0] / image_info['width']
                box[1] = box[1] / image_info['height']
                grounding_boxes[j, :] = box
                j = j + 1
        # self.vis_item(fpn_data['image'], grounding_instances, idx)

        grounding_instances = {'gt': grounding_instances, 'gt_box': grounding_boxes}

        return caption, grounding_instances, \
            ann_categories, ann_types, noun_vector_padding, ret_noun_vector, fpn_data

class PanopticNarrativeGroundingValDataset(Dataset):
    """Panoptic Narrative Grounding dataset."""

    def __init__(self, cfg, split, train=True):
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

        self.mask_transform = Resize((256, 256))

        self.ann_dir = osp.join(cfg.data_path, "annotations")
        self.panoptic = self.load_json(
            osp.join(self.ann_dir, "panoptic_{:s}.json".format(split))
        )
        self.images = self.panoptic["images"]
        self.images = {i["id"]: i for i in self.images}
        self.panoptic_anns = self.panoptic["annotations"]
        self.panoptic_anns = {a["image_id"]: a for a in self.panoptic_anns}
        if not osp.exists(
            osp.join(self.ann_dir, 
                "png_coco_{:s}_dataloader.json".format(split),)
        ):
            print("No such a dataset")
        else:
            self.panoptic_narrative_grounding = self.load_json(
                osp.join(self.ann_dir, 
                    "png_coco_{:s}_dataloader.json".format(split),)
            )
        self.panoptic_narrative_grounding = [
            ln
            for ln in self.panoptic_narrative_grounding
            if (
                torch.tensor(np.array([item for sublist in ln["labels"] 
                    for item in sublist]))
                != -2
            ).any()
        ]
        fpn_dataset, self.fpn_mapper = fpn_data(cfg, split[:-4])
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
        return len(self.panoptic_narrative_grounding)
    
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
        localized_narrative = self.panoptic_narrative_grounding[idx]
        caption = localized_narrative['caption']
        image_id = int(localized_narrative['image_id'])
        fpn_data = self.fpn_mapper(self.fpn_dataset[image_id])  
        image_info = self.images[image_id]
        labels = localized_narrative['labels']

        noun_vector = localized_narrative['noun_vector']
        if len(noun_vector) > (self.cfg.max_sequence_length - 2):
            noun_vector_padding = \
                    noun_vector[:(self.cfg.max_sequence_length - 2)]
        elif len(noun_vector) < (self.cfg.max_sequence_length - 2): 
            noun_vector_padding = \
                noun_vector + [0] * (self.cfg.max_sequence_length - \
                    2 - len(noun_vector))
        noun_vector_padding = [0] + noun_vector_padding + [0]
        noun_vector_padding = torch.tensor(np.array(noun_vector_padding)).long()
        assert len(noun_vector_padding) == \
            self.cfg.max_sequence_length
        
        ret_noun_vector = noun_vector_padding[noun_vector_padding.nonzero()].flatten()
        assert len(ret_noun_vector) <= self.cfg.max_seg_num
        if len(ret_noun_vector) < self.cfg.max_seg_num:
            ret_noun_vector = torch.cat([ret_noun_vector, \
                ret_noun_vector.new_zeros((self.cfg.max_seg_num - len(ret_noun_vector)))])
        cur_phrase_index = ret_noun_vector[ret_noun_vector!=0]
                
        _, cur_index_counts = torch.unique_consecutive(cur_phrase_index, return_counts=True)
        cur_phrase_interval = torch.cumsum(cur_index_counts, dim=0)
        cur_phrase_interval = torch.cat([cur_phrase_interval.new_zeros((1)), cur_phrase_interval])
        # ret_noun_vector: [max_seg_num]

        ann_types = [0] * len(labels)
        for i, l in enumerate(labels):
            l = torch.tensor(np.array(l))
            if (l != -2).any():
                ann_types[i] = 1 if (l != -2).sum() == 1 else 2
        ann_types = torch.tensor(np.array(ann_types)).long()
        ann_types = ann_types[ann_types.nonzero()].flatten()
        assert len(ann_types) <= self.cfg.max_seg_num
        if len(ann_types) < self.cfg.max_seg_num:
            ann_types = torch.cat([ann_types, \
                ann_types.new_zeros((self.cfg.max_seg_num - len(ann_types)))])
        ann_types_valid = ann_types.new_zeros(self.cfg.max_phrase_num)
        ann_types_valid[:len(cur_phrase_interval)-1] = ann_types[cur_phrase_interval[:-1]]
        
        ann_categories = torch.zeros([
            self.cfg.max_phrase_num]).long()
        panoptic_ann = self.panoptic_anns[image_id]
        panoptic_segm = io.imread(
            osp.join(
                self.ann_dir,
                "panoptic_segmentation",
                self.split,
                "{:012d}.png".format(image_id),
            )
        )
        panoptic_segm = (
            panoptic_segm[:, :, 0]
            + panoptic_segm[:, :, 1] * 256
            + panoptic_segm[:, :, 2] * 256 ** 2
        )
        grounding_instances = torch.zeros(
            [self.cfg.max_phrase_num, image_info['height'], image_info['width']]
        )
        grounding_boxes = torch.zeros(
            [self.cfg.max_seg_num, 4]
        )
        j = 0
        k = 0
        for i, bbox in enumerate(localized_narrative["boxes"]):
            flag = False
            for b in bbox:
                if b != [0] * 4:
                    flag = True
            if not flag:
                continue
            
            for b in bbox:
                box = torch.zeros(4)
                box[0] = image_info['width']
                box[1] = image_info['height']
                if b != [0] * 4:
                    flag = True
                    segment_info = [
                        s for s in panoptic_ann["segments_info"] 
                        if s["bbox"] == b
                    ][0]
                    segment_cat = [
                        c
                        for c in self.panoptic["categories"]
                        if c["id"] == segment_info["category_id"]
                    ][0]
                    instance = torch.zeros([image_info['height'],
                            image_info['width']])
                    instance[panoptic_segm == segment_info["id"]] = 1

                    if j in cur_phrase_interval[:-1]:
                        grounding_instances[k, :] += instance
                        
                        if box[0] > b[0]:
                            box[0] = b[0]
                        if box[1] > b[1]:
                            box[1] = b[1]
                        if box[2] < b[0] + b[2]:
                            box[2] = b[0] + b[2]
                        if box[3] < b[1] + b[3]:
                            box[3] = b[1] + b[3]

                        ann_categories[k] = 1 if \
                                segment_cat["isthing"] else 2
            if j in cur_phrase_interval[:-1]:
                box[2] = (box[2]-box[0]) / image_info['width']
                box[3] = (box[3]-box[1]) / image_info['height']
                box[0] = box[0] / image_info['width']
                box[1] = box[1] / image_info['height']
                grounding_boxes[k, :] = box
                k = k + 1   
            j = j + 1
        assert k == len(cur_phrase_interval) - 1
        # self.vis_item(fpn_data['image'], grounding_instances, idx)
        grounding_instances = {'gt': grounding_instances, 'gt_box': grounding_boxes}
        ret_noun_vector = {'inter': cur_phrase_interval}

        return caption, grounding_instances, \
            ann_categories, ann_types_valid, noun_vector_padding, ret_noun_vector, fpn_data, ann_types

class Flickr30KDataset(Dataset):
    def __init__(self, cfg,  train=True):
        """
        Args:
            Args:
            cfg (CfgNode): configs.
            train (bool):
        """
        self.cfg = cfg
        self.train = train # True or False
        # split = 'val2017'
        # train2017 or val2017
        self.tokenzier = BertTokenizer.from_pretrained('/root/NICE/pretrained_models/bert/bert-base-uncased.txt', do_lower_case=True)
        self.mask_transform = Resize((256, 256)) 

        self.image_list = []
        if self.train:
            with open("/data/flickr30k/flickr30k_entities/train.txt", 'r') as f:
                for line in f.readlines():
                    line = line.strip('\n')
                    self.image_list.append(line)
        else:
            with open("/data/flickr30k/flickr30k_entities/test.txt", 'r') as f:
                for line in f.readlines():
                    line = line.strip('\n')
                    self.image_list.append(line)

        self.image_root = '/data/flickr30k/flickr30k-images'
        self.annotations_root = '/data/flickr30k/flickr30k_entities/annotations/Annotations'
        self.sentence_root = '/data/flickr30k/flickr30k_entities/annotations/Sentences'

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

    def get_sentence_data(self, fn):
        """
        Parses a sentence file from the Flickr30K Entities dataset

        input:
        fn - full file path to the sentence file to parse
        
        output:
        a list of dictionaries for each sentence with the following fields:
            sentence - the original sentence
            phrases - a list of dictionaries for each phrase with the
                        following fields:
                        phrase - the text of the annotated phrase
                        first_word_index - the position of the first word of
                                            the phrase in the sentence
                        phrase_id - an identifier for this phrase
                        phrase_type - a list of the coarse categories this 
                                        phrase belongs to

        """
        with open(fn, 'r') as f:
            sentences = f.read().split('\n')

        annotations = []
        for sentence in sentences:
            if not sentence:
                continue

            first_word = []
            phrases = []
            phrase_id = []
            phrase_type = []
            words = []
            current_phrase = []
            add_to_phrase = False
            for token in sentence.split():
                if add_to_phrase:
                    if token[-1] == ']':
                        add_to_phrase = False
                        token = token[:-1]
                        current_phrase.append(token)
                        phrases.append(' '.join(current_phrase))
                        current_phrase = []
                    else:
                        current_phrase.append(token)

                    words.append(token)
                else:
                    if token[0] == '[':
                        add_to_phrase = True
                        first_word.append(len(words))
                        parts = token.split('/')
                        phrase_id.append(parts[1][3:])
                        phrase_type.append(parts[2:])
                    else:
                        words.append(token)

            sentence_data = {'sentence' : ' '.join(words), 'phrases' : []}
            for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id, phrase_type):
                sentence_data['phrases'].append({'first_word_index' : index,
                                                'phrase' : phrase,
                                                'phrase_id' : p_id,
                                                'phrase_type' : p_type})

            annotations.append(sentence_data)

        return annotations

    def get_annotations(self, fn):
        """
        Parses the xml files in the Flickr30K Entities dataset

        input:
        fn - full file path to the annotations file to parse

        output:
        dictionary with the following fields:
            scene - list of identifiers which were annotated as
                    pertaining to the whole scene
            nobox - list of identifiers which were annotated as
                    not being visible in the image
            boxes - a dictionary where the fields are identifiers
                    and the values are its list of boxes in the 
                    [xmin ymin xmax ymax] format
        """
        tree = ET.parse(fn)
        root = tree.getroot()
        size_container = root.findall('size')[0]
        anno_info = {'boxes' : {}, 'scene' : [], 'nobox' : []}
        for size_element in size_container:
            anno_info[size_element.tag] = int(size_element.text)

        for object_container in root.findall('object'):
            for names in object_container.findall('name'):
                box_id = names.text
                box_container = object_container.findall('bndbox')
                if len(box_container) > 0:
                    if box_id not in anno_info['boxes']:
                        anno_info['boxes'][box_id] = []
                    xmin = int(box_container[0].findall('xmin')[0].text) - 1
                    ymin = int(box_container[0].findall('ymin')[0].text) - 1
                    xmax = int(box_container[0].findall('xmax')[0].text) - 1
                    ymax = int(box_container[0].findall('ymax')[0].text) - 1
                    anno_info['boxes'][box_id].append([xmin, ymin, xmax, ymax])
                else:
                    nobndbox = int(object_container.findall('nobndbox')[0].text)
                    if nobndbox > 0:
                        anno_info['nobox'].append(box_id)

                    scene = int(object_container.findall('scene')[0].text)
                    if scene > 0:
                        anno_info['scene'].append(box_id)

        return anno_info

    def convert_sents_to_features(self, sent, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        tokens_a = tokenizer.tokenize(sent.strip()) #list of words
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2: #max_seq_length=230
            tokens_a = tokens_a[:(max_seq_length - 2)] 
        
        # Keep segment id which allows loading BERT-weights.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        if max_seq_length == 231:
            return input_ids

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding

        assert len(input_ids) == max_seq_length

        return input_ids

    def strStr(self, haystack, needle) -> int:
        def KMP(s, p):
            nex = getNext(p)
            i = 0
            j = 0   # 分别是s和p的指针
            while i < len(s) and j < len(p):
                if j == -1 or s[i] == p[j]: # j==-1是由于j=next[j]产生
                    i += 1
                    j += 1
                else:
                    j = nex[j]

            if j == len(p): # j走到了末尾，说明匹配到了
                return i - j
            else:
                return -1

        def getNext(p):
            nex = [0] * (len(p) + 1)
            nex[0] = -1
            i = 0
            j = -1
            while i < len(p):
                if j == -1 or p[i] == p[j]:
                    i += 1
                    j += 1
                    nex[i] = j     # 这是最大的不同：记录next[i]
                else:
                    j = nex[j]

            return nex
        
        return KMP(haystack, needle)

    def __len__(self):
        return len(self.image_list)
    
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
        image_id = self.image_list[idx]

        image_path = osp.join(self.image_root, image_id+'.jpg')
        annotation_path = osp.join(self.annotations_root, image_id+'.xml')
        sentence_path = osp.join(self.sentence_root, image_id+'.txt')

        annotation = self.get_annotations(annotation_path)
        sentences = self.get_sentence_data(sentence_path)
        image = torch.from_numpy(np.asarray(Image.open(image_path).convert('RGB')).copy()).permute(2, 0, 1)

        fpn_data = {
            'filename': image_path,
            'height': annotation['height'],
            'width': annotation['width'],
            'image': image
        }

        for boxes in annotation['boxes']:
            box = torch.zeros(4)
            box[0] = annotation['width']
            box[1] = annotation['height']
            for b in annotation['boxes'][boxes]:
                if box[0] > b[0]:
                    box[0] = b[0]
                if box[1] > b[1]:
                    box[1] = b[1]
                if box[2] < b[2]:
                    box[2] = b[2]
                if box[3] < b[3]:
                    box[3] = b[3]
            
            # box[2] = box[2] - box[0]
            # box[3] = box[3] - box[1]
            box[0] = box[0] / annotation['width']
            box[1] = box[1] / annotation['height']
            box[2] = box[2] / annotation['width']
            box[3] = box[3] / annotation['height']
            annotation['boxes'][boxes] = box

        sentence = random.choice(sentences)

        caption = sentence['sentence']
        caption_token = self.convert_sents_to_features(caption, self.cfg.max_sequence_length, self.tokenzier)
        noun_vectors = torch.zeros(self.cfg.max_sequence_length)

        gt_boxes = torch.zeros(self.cfg.max_seg_num, 4)
        for i, phrase in enumerate(sentence['phrases']):

            if phrase['phrase_id'] in annotation['boxes']:

                noun_vector = torch.zeros(self.cfg.max_sequence_length)
                phrase_token = self.convert_sents_to_features(phrase['phrase'], 231, self.tokenzier)[1: -1]
                first_index = self.strStr(caption_token, phrase_token)
                noun_vector[first_index: first_index+len(phrase_token)] = i+1

                noun_vectors += noun_vector
                gt_boxes[i] = annotation['boxes'][phrase['phrase_id']]
        
        return fpn_data, noun_vectors, gt_boxes, caption

class Flickr30KProDataset(Dataset):
    def __init__(self, cfg,  train=True):
        """
        Args:
            Args:
            cfg (CfgNode): configs.
            train (bool):
        """
        self.cfg = cfg
        self.train = train # True or False
        # split = 'val2017'
        # train2017 or val2017
        self.tokenzier = BertTokenizer.from_pretrained('/root/NICE/pretrained_models/bert/bert-base-uncased.txt', do_lower_case=True)
        self.mask_transform = Resize((256, 256)) 

        if self.train:

            self.annotation_list = torch.load('/data/flickr30k/flickr_train.pth')
        else:

            self.annotation_list = torch.load('/data/flickr30k/flickr_test.pth')

        self.image_root = '/data/flickr30k/flickr30k-images'


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

    def convert_sents_to_features(self, sent, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        tokens_a = tokenizer.tokenize(sent.strip()) #list of words
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2: #max_seq_length=230
            tokens_a = tokens_a[:(max_seq_length - 2)] 
        
        # Keep segment id which allows loading BERT-weights.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        if max_seq_length == 231:
            return input_ids

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding

        assert len(input_ids) == max_seq_length

        return input_ids

    def strStr(self, haystack, needle) -> int:
        def KMP(s, p):
            nex = getNext(p)
            i = 0
            j = 0   # 分别是s和p的指针
            while i < len(s) and j < len(p):
                if j == -1 or s[i] == p[j]: # j==-1是由于j=next[j]产生
                    i += 1
                    j += 1
                else:
                    j = nex[j]

            if j == len(p): # j走到了末尾，说明匹配到了
                return i - j
            else:
                return -1

        def getNext(p):
            nex = [0] * (len(p) + 1)
            nex[0] = -1
            i = 0
            j = -1
            while i < len(p):
                if j == -1 or p[i] == p[j]:
                    i += 1
                    j += 1
                    nex[i] = j     # 这是最大的不同：记录next[i]
                else:
                    j = nex[j]

            return nex
        
        return KMP(haystack, needle)

    def __len__(self):
        return len(self.annotation_list)
    
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
        annotation = self.annotation_list[idx]

        image_id = annotation[0]

        image_path = osp.join(self.image_root, image_id)


        image = torch.from_numpy(np.asarray(Image.open(image_path).convert('RGB')).copy()).permute(2, 0, 1)

        height, width = image.shape[-2], image.shape[-1]

        fpn_data = {
            'filename': image_path,
            'height': height,
            'width': width,
            'image': image
        }

        caption = annotation[-1]
        caption_token = self.convert_sents_to_features(caption, self.cfg.max_sequence_length, self.tokenzier)
        noun_vectors = torch.zeros(self.cfg.max_sequence_length)

        gt_boxes = torch.zeros(self.cfg.max_seg_num, 4)
        for i, phrase in enumerate(annotation[3]):

            noun_vector = torch.zeros(self.cfg.max_sequence_length)
            phrase_token = self.convert_sents_to_features(phrase, 231, self.tokenzier)[1: -1]
            first_index = self.strStr(caption_token, phrase_token)
            noun_vector[first_index: first_index+len(phrase_token)] = i+1

            noun_vectors += noun_vector
            gt_boxes[i] = torch.from_numpy(annotation[2][i])
        
        gt_boxes[:, 0] = gt_boxes[:, 0] / width
        gt_boxes[:, 1] = gt_boxes[:, 1] / height
        gt_boxes[:, 2] = gt_boxes[:, 2] / width
        gt_boxes[:, 3] = gt_boxes[:, 3] / height    
        return fpn_data, noun_vectors, gt_boxes, caption