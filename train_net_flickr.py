# Standard lib imports
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import time
import numpy as np
import os.path as osp
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from detectron2.structures import ImageList

# PyTorch imports
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist

# Local imports
from utils.meters import average_accuracy
from utils import AverageMeter
from utils import compute_mask_IoU, compute_box_IoU
from data import Flickr30KDataset, Flickr30KProDataset
# from models.knet.knet import KNet
from models.NICE.MainModule import MainModule
from models.dice_loss import DiceLoss
from models.cross_entropy_loss import CrossEntropyLoss
from models.L1_loss import SmoothL1Loss
from models.Giou_loss import GIoULoss
from models.encoder_bert import BertEncoder
from utils.logger import setup_logger
from utils.collate_fn import default_collate
from utils.distributed import (all_gather, all_reduce)
from models.extract_fpn_with_ckpt_load_from_detectron2 import fpn
from utils.contrastive import CKDLoss
import time

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def train_epoch(train_loader, bert_encoder, fpn_model, model, 
    optimizer, epoch, cfg, logger, writer):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): train loader.
        model (model): the model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        loss_functions (loss): the loss function to optimize.
        epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            PNG/config/defaults.py
    """
    if dist.get_rank()==0:
        logger.info('-' * 89)
        logger.info('Training epoch {:5d}'.format(epoch))
        logger.info('-' * 89)

    # Enable train mode.
    model.train()
    if cfg.bert_freeze:
        bert_encoder.eval()
    else:
        bert_encoder.train()
    if cfg.fpn_freeze:
        fpn_model.eval()
    else:
        fpn_model.train()

    epoch_loss = AverageMeter()
    time_stats = AverageMeter()

    # Use cuda if available
    dice_loss = DiceLoss()
    ce_loss = CrossEntropyLoss(use_sigmoid=True)
    l1_loss = SmoothL1Loss()
    giou_loss = GIoULoss()
    # closs, c2loss = [], []
    # for i in range(cfg.num_stages):
    #     closs.append(CKDLoss())
        # c2loss.append(CKDLoss())

    for (batch_idx, (fpn_input_data, noun_vector, gts, caption)) in enumerate(train_loader):

        start_time = time.time()
        
        lang_feat = bert_encoder(caption) #bert for caption
        lang_feat_valid = lang_feat.new_zeros((lang_feat.shape[0], \
            cfg.max_seg_num, lang_feat.shape[-1]))
        
        ann_types = lang_feat.new_zeros(lang_feat.shape[0], \
            cfg.max_seg_num)
        
        for i in range(len(lang_feat)):
            cur_lang_feat = lang_feat[i][noun_vector[i].nonzero().flatten()]
            lang_feat_valid[i, :cur_lang_feat.shape[0], :] = cur_lang_feat
            ann_types[i, :cur_lang_feat.shape[0]] = 1
            
        fpn_feature = fpn_model(fpn_input_data) #fpn for imgs

        # preprocessing for gt masks
        
        # gts: [B, max_seg_num, H//4, W//4]
        # lang_feat_valid: [B, max_seg_num, C]
        # predictions, kernels, gt_ins_feats = model(fpn_feature, lang_feat_valid, gts=gts) #Knet

        _, _, predictions_box = model(fpn_feature, lang_feat_valid) #Knet

        end_time = time.time()

        x = end_time - start_time

        loss = 0
        grad_sample = ann_types != 0
        pred_box = predictions_box[0][grad_sample]
        gt_box = torch.zeros(pred_box.shape).to(cfg.local_rank)
        # with torch.no_grad():

        #     gts_box = [gts[i].to(cfg.local_rank) for i in range(len(gts))]
        #     gts_box = torch.stack(gts_box, dim=0)
        first_index = 0
        for i in range(len(ann_types)):
            max_ = int(torch.max(noun_vector[i]))

            for nums in range(1, max_+1):
                temp_length = sum(noun_vector[i]==nums)
                gt_box[first_index: first_index+temp_length, :] = gts[i, nums-1]
                first_index += temp_length

        # constrative_loss =  SAL(output, text, gts, grad_sample)

        # loss = constrative_loss + 2 * ce_loss(predictions[-1][grad_sample], gt) + 2 * dice_loss(predictions[-1][grad_sample], gt)

        for pred_box in predictions_box:
            loss += l1_loss(pred_box[grad_sample], gt_box).mean() + giou_loss(pred_box[grad_sample], gt_box).mean()
        

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()

        # Update the parameters.
        optimizer.step()

        # Gather all the predictions across all the devices.
        if cfg.num_gpus > 1:
            loss = all_reduce([loss])[0]
        
        time_stats.update(time.time() - start_time, 1)
        epoch_loss.update(loss.item(), 1)

        if dist.get_rank()==0:
            if (batch_idx % cfg.log_period == 0):
                elapsed_time = time_stats.avg
                logger.info(' [{:5d}] ({:5d}/{:5d}) | ms/batch {:.4f} |'
                    ' avg loss {:.6f} |'
                    ' lr {:.7f} |'.format(
                        epoch, batch_idx, len(train_loader),
                        elapsed_time * 1000, 
                        epoch_loss.avg,
                        optimizer.param_groups[0]["lr"]))
                writer.add_scalar('train/loss', epoch_loss.avg, epoch * len(train_loader) + batch_idx)
                writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], epoch * len(train_loader) + batch_idx)
                writer.flush()

    return epoch_loss.avg

def upsample_eval(tensors, pad_value=0, t_size=[400, 400]):
    batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + list(t_size)
    batched_imgs = tensors[0].new_full(batch_shape, pad_value)
    for img, pad_img in zip(tensors, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
    return batched_imgs


@torch.no_grad()
def evaluate(val_loader, bert_encoder, fpn_model, model, epoch, cfg, logger, writer):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        cfg (CfgNode): configs. Details can be found in
            PNG/config/defaults.py
    """
    if dist.get_rank()==0:
        logger.info('-' * 89)
        logger.info('Evaluation on val set epoch {:5d}'.format(epoch))
        logger.info('-' * 89)
    
    # Enable eval mode.
    model.eval()
    bert_encoder.eval()
    fpn_model.eval()
    
    time_status = AverageMeter()
    boxes_iou = []

    # pbar = tqdm(total=len(val_loader))
    for (batch_idx, (fpn_input_data, noun_vector, gts, caption)) in enumerate(val_loader):
        # ret_noun_vector = ret_noun_vector.to(cfg.local_rank)
        noun_vector = noun_vector.cuda()
        # Perform the forward pass
        with torch.no_grad():
            lang_feat = bert_encoder(caption) #bert for caption
            lang_feat_valid = lang_feat.new_zeros((lang_feat.shape[0], \
                cfg.max_seg_num, lang_feat.shape[-1]))
            
            ann_types = lang_feat.new_zeros(lang_feat.shape[0], \
                cfg.max_seg_num)
            
            for i in range(len(lang_feat)):
                cur_lang_feat = lang_feat[i][noun_vector[i].nonzero().flatten()]
                lang_feat_valid[i, :cur_lang_feat.shape[0], :] = cur_lang_feat
                
            fpn_feature = fpn_model(fpn_input_data) #fpn for imgs
            start_time = time.time()
            _, _, predictions_boxes = model(fpn_feature, lang_feat_valid)
            end_time = time.time()

            predictions_boxes = predictions_boxes[-1]
            x = end_time - start_time
            time_status.update(x, 1)

            predictions_valid_boxes = predictions_boxes.new_zeros((predictions_boxes.shape[0], cfg.max_phrase_num, \
                predictions_boxes.shape[-1]))
            
            for i in range(len(predictions_boxes)):
                max_ = int(torch.max(noun_vector[i]))
                ann_types[i, :max_] = 1
                cur_phrase_interval = []
                first_index = 0
                cur_phrase_interval.append(first_index)
                for nums in range(1, max_+1):
                    first_index += sum(noun_vector[i]==nums)
                    cur_phrase_interval.append(first_index.item())
                for j in range(len(cur_phrase_interval)-1):
                    for k in range(cur_phrase_interval[j], cur_phrase_interval[j+1]):
                        predictions_valid_boxes[i, j, :] = predictions_valid_boxes[i, j, :] + predictions_boxes[i][k]
                    predictions_valid_boxes[i, j, :] = predictions_valid_boxes[i, j, :] / (cur_phrase_interval[j+1]-cur_phrase_interval[j])

        # preprocessing for gt masks
        with torch.no_grad():

            gts_box = [gts[i].to(cfg.local_rank) for i in range(len(gts))]
            gts_box = torch.stack(gts_box, dim=0)
        
        # Gather all the predictions across all the devices.
        if cfg.num_gpus > 1:
            predictions_valid_boxes, gts_box, ann_types, predictions_boxes, noun_vector = all_gather(
                [predictions_valid_boxes, gts_box, ann_types, predictions_boxes, noun_vector]
            )

        # Evaluation

        for pb, tb, s, vector, pvb  in zip(predictions_valid_boxes, gts_box, ann_types, noun_vector, predictions_boxes):
            for i in range(cfg.max_phrase_num):
                if s[i] == 0:
                    continue
                else:
                    pd_b = pb[i]

                    _, _, box_iou = compute_box_IoU(pd_b, tb[i])
                    cur_phrase_interval = []
                    first_index = 0
                    max_ = int(torch.max(vector))
                    cur_phrase_interval.append(first_index)
                    for nums in range(1, max_+1):
                        first_index += sum(vector==nums)
                        cur_phrase_interval.append(first_index.item())
                    for j in range(len(cur_phrase_interval)-1):
                        for k in range(cur_phrase_interval[j], cur_phrase_interval[j+1]):
                            _, _, temp_iou = compute_box_IoU(pvb[k], tb[i])
                            if box_iou < temp_iou:
                                box_iou = temp_iou

                    boxes_iou.append(box_iou.cpu().item())
        
        if batch_idx % 100 == 0:
            print(f'{batch_idx}/{len(val_loader)}')
        # if dist.get_rank()==0:
        #     pbar.update(1)
            # if batch_idx % cfg.log_period == 0:
            # tqdm.write('acc@0.5: {:.5f} | AA: {:.5f}'.format(accuracy_score(np.ones([len(instances_iou)]), np.array(instances_iou) > 0.5), average_accuracy(instances_iou))) 
    
    # pbar.close()
    # Final evaluation metrics


    box_AA = average_accuracy(boxes_iou, save_fig=cfg.save_fig, output_dir=cfg.output_dir, filename='box_overall')
    box_accuracy = accuracy_score(np.ones([len(boxes_iou)]), np.array(boxes_iou) > 0.5)
    if dist.get_rank()==0:


        logger.info('| final box_acc@0.5: {:.5f} | final box_AA: {:.5f} | inferenct time: {:.5f}'.format(
                                            box_accuracy,
                                            box_AA,
                                            time_status.avg))

        writer.add_scalar('aa/box_acc@0.5', box_accuracy, epoch)
        writer.add_scalar('aa/box_final', box_AA, epoch)
        
    return box_AA



def train(cfg):
    local_rank = cfg.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=cfg.backend)

    if dist.get_rank() == 0:
        logger = setup_logger(cfg.output_dir, dist.get_rank())
        writer = SummaryWriter(osp.join(cfg.output_dir, 'tensorboard'))
    else:
        logger, writer = None, None

    # Set random seed from configs.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    if dist.get_rank() == 0:
        logger.info(cfg)

    bert_encoder = BertEncoder(cfg).to(local_rank)
    bert_encoder = DDP(bert_encoder, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True) 
    fpn_model = fpn(cfg.detectron2_ckpt, cfg.detectron2_cfg)
    fpn_model = fpn_model.to(local_rank)
    fpn_model = DDP(fpn_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    model = MainModule().to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    ema = EMA(model.module, 0.999)
    ema.register()

    if cfg.bert_freeze:
        cnt = 0
        for n, c in bert_encoder.named_parameters():
            c.requires_grad = False
            cnt += 1
        if dist.get_rank() == 0:
            logger.info(f'Freezing {cnt} parameters of BERT.')

    if cfg.fpn_freeze:
        cnt = 0
        for n, c in fpn_model.named_parameters():
            c.requires_grad = False
            cnt += 1
        if dist.get_rank() == 0:
            logger.info(f'Freezing {cnt} parameters of FPN.')

    if not cfg.test_only:
        train_dataset = Flickr30KProDataset(cfg, True)
        distributed_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            sampler = distributed_sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            collate_fn=default_collate,
        )

    val_dataset = Flickr30KProDataset(cfg, False)
    distributed_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = DataLoader(
        val_dataset,
        batch_size=12,
        sampler = distributed_sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=default_collate,
    )

    if cfg.bert_freeze and cfg.fpn_freeze:
        # train_params += list(filter(lambda p: p.requires_grad, bert_encoder.parameters()))
        # train_params += list(filter(lambda p: p.requires_grad, fpn_model.parameters()))
        train_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        if dist.get_rank() == 0:
            logger.info(f'{len(train_params)} training params.')
        optimizer = optim.Adam(train_params,
                            lr=cfg.base_lr, weight_decay=cfg.weight_decay)
    elif cfg.fpn_freeze:
        bert_encoder_params = list(filter(lambda p: p.requires_grad, bert_encoder.parameters()))
        model_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = optim.Adam([{'params': model_params, 'lr':cfg.base_lr},
                               {'params': bert_encoder_params, 'lr':cfg.base_lr/10}])
    else:
        fpn_params = list(filter(lambda p: p.requires_grad, fpn_model.parameters()))
        bert_encoder_params = list(filter(lambda p: p.requires_grad, bert_encoder.parameters()))
        model_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = optim.Adam([{'params': model_params, 'lr':cfg.base_lr},
                               {'params': fpn_params, 'lr':cfg.base_lr/10},
                               {'params': bert_encoder_params, 'lr':cfg.base_lr/10}])
    
    if cfg.scheduler == 'step':
        if not cfg.fpn_freeze and not cfg.bert_freeze:
            milestones = [10, 20, 30]
            lambda1 = lambda epoch: 1 if epoch < milestones[0] else 0.5 if epoch < milestones[1] else 0.25 if epoch < milestones[2] else 0.125
            lambda2 = lambda epoch: 1
            lambda3 = lambda epoch: 1
            lambda_list = [lambda1, lambda2, lambda3]
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_list)
        else:
            milestones = [10, 12, 14]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, \
                                                      gamma=0.5)
    elif cfg.scheduler == 'reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, \
                                                               mode='max', min_lr=1e-6, \
                                                               patience=2)
    else:
        raise ValueError(f'{cfg.scheduler} NOT IMPLEMENT!!!')


    start_epoch, best_val_score = 0, None
    if osp.exists(cfg.ckpt_path):
        if dist.get_rank()==0:
            print('Loading model from: {0}'.format(cfg.ckpt_path))
        checkpoint = torch.load(cfg.ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint['model_state'])
        fpn_model.load_state_dict(checkpoint['fpn_model_state'])
        bert_encoder.load_state_dict(checkpoint['bert_model_state'])
        start_epoch = checkpoint['epoch'] + 1
        # optimizer.load_state_dict(checkpoint['optimizer_state'])
        # scheduler.load_state_dict(checkpoint['scheduler_state'])
        best_val_score = checkpoint['best_val_score']

    if cfg.test_only:
        epoch = 0
        evaluate(val_loader, bert_encoder, \
        fpn_model, model, epoch, cfg, logger, writer)
        return

    if dist.get_rank()==0:
        logger.info('Train begins...')

    # Perform the training loop
    for epoch in range(start_epoch, cfg.epoch):
        epoch_start_time = time.time()
        # Shuffle the dataset
        train_loader.sampler.set_epoch(epoch)
        # Train for one epoch
        train_loss = train_epoch(train_loader, bert_encoder, \
            fpn_model, model, optimizer, epoch, cfg, logger, writer)
        
        ema.update()
        ema.apply_shadow()
        accuracy = evaluate(val_loader, bert_encoder, \
            fpn_model, model, epoch, cfg, logger, writer)
        if dist.get_rank() == 0:
            writer.flush()

        if cfg.scheduler == 'step':
            scheduler.step()
        elif cfg.scheduler == 'reduce':
            scheduler.step(accuracy)
        else:
            raise ValueError(f'{cfg.scheduler} NOT IMPLEMENT!!!')

        if dist.get_rank()==0:
            # Save best model in the validation set
            if best_val_score is None or accuracy > best_val_score:
                best_val_score = accuracy
                model_final_path = osp.join(cfg.output_dir, 'model_best.pth')
                model_final = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "fpn_model_state": fpn_model.state_dict(),
                    "bert_model_state": bert_encoder.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val_score": accuracy
                }
                torch.save(model_final, model_final_path)
            if epoch > cfg.save_ckpt:
                model_final_path = osp.join(cfg.output_dir, f'checkpoint_{epoch}.pth')
            else:
                model_final_path = osp.join(cfg.output_dir, f'checkpoint.pth')

            model_final = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "fpn_model_state": fpn_model.state_dict(),
                "bert_model_state": bert_encoder.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_score": accuracy
            }
            torch.save(model_final, model_final_path)
            logger.info('-' * 89)
            logger.info('| end of epoch {:3d} | time: {:5.2f}s '
                    '| epoch loss {:.6f} |'.format(
                        epoch, time.time() - epoch_start_time, train_loss))
            logger.info('-' * 89)
    if dist.get_rank() == 0:
        writer.close()
