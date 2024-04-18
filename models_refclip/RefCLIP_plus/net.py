# coding=utf-8

import torch
import torch.nn as nn

from models_refclip.language_encoder import language_encoder
from models_refclip.visual_encoder import visual_encoder
from models_refclip.RefCLIP_plus.head import WeakREChead
from models_refclip.network_blocks import MultiScaleFusion
import sys
from utils.general import non_max_suppression, xywh2xyxy
from utils.segment.general import batch_process_mask



class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size):
        super(Net, self).__init__()
        self.select_num = __C.SELECT_NUM
        self.visual_encoder = visual_encoder(__C).eval()
        self.lang_encoder = language_encoder(__C, pretrained_emb, token_size)

        self.linear_vs = nn.Linear(1024, __C.HIDDEN_SIZE)
        self.linear_ts = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.head = WeakREChead(__C)
        
        if "yolov5x" in __C.PRETRAIN_WEIGHT:
            self.multi_scale_manner = MultiScaleFusion(v_planes=(320, 640, 1280), hiden_planes=1024, scaled=True)
        elif "yolov5l" in __C.PRETRAIN_WEIGHT:
            self.multi_scale_manner = MultiScaleFusion(v_planes=(256, 512, 1024), hiden_planes=1024, scaled=True)
        elif "yolov5m" in __C.PRETRAIN_WEIGHT:
            self.multi_scale_manner = MultiScaleFusion(v_planes=(192, 384, 768), hiden_planes=1024, scaled=True)

        self.class_num = __C.CLASS_NUM
        if __C.VIS_FREEZE:
            self.frozen(self.visual_encoder)

    def frozen(self, module):
        if getattr(module, 'module', False):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, x, y, box=None,wei = None):

        # Vision and Language Encoding
        _, _, ih, iw = x.shape
        with torch.no_grad():
            boxes_all, x_, boxes_sml, protos, masks_in_list = self.visual_encoder(x)  # preds: boxes_all
        y_ = self.lang_encoder(y)

        # Vision Multi Scale Fusion
        s, m, l = x_
        x_input = [l, m, s]
        l_new, m_new, s_new = self.multi_scale_manner(x_input)
        x_ = [s_new, m_new, l_new]

        # Anchor Selection
        boxes_sml_new = []
        masks_sml_new = []
        mean_i = torch.mean(boxes_sml[0], dim=2, keepdim=True)
        mean_i = mean_i.squeeze(2)[:, :, 4]
        vals, indices = mean_i.topk(k=int(self.select_num), dim=1, largest=True, sorted=True)
        bs, gridnum, anncornum, ch = boxes_sml[0].shape
        if len(masks_in_list) > 0:
            _, _, _, ch_m = masks_in_list[0].shape
        
        
        
        last_indices = None
        if self.training and box is not None:
            xy_c = box.squeeze(1)[:, :2]
            xy_c = (xy_c * 13).floor()
            tar_c = xy_c[:, 1] * 13 + xy_c[:, 0]
            # tar_c = torch.full(tar_c.shape, 0).to(tar_c.device)
            import warnings
            if tar_c.min() < 0:
                warnings.warn('Some elements in `tar_c` are smaller than 0. They will be replaced with 0.')
                tar_c[tar_c < 0] = 0
            if tar_c.max() > 168:
                warnings.warn('Some elements in `tar_c` are larger than 168. They will be replaced with 168.')
                tar_c[tar_c > 168] = 168
            tar_c = tar_c.long().unsqueeze(1)
            indices = torch.where(torch.eq(indices, tar_c), indices[:, -1].unsqueeze(1).repeat(1, indices.shape[1]),
                                  indices)
            indices[:, -1] = tar_c.squeeze(1)
            sorted_x, indices_temp = torch.sort(indices, dim=1)
            last_indices = torch.where(indices_temp == indices.shape[1] - 1)[1]



        bs_, selnum = indices.shape
        box_sml_new = boxes_sml[0].masked_select(
            torch.zeros(bs, gridnum).to(boxes_sml[0].device).scatter(1, indices, 1).bool().unsqueeze(2).unsqueeze(
                3).expand(bs, gridnum, anncornum, ch)).contiguous().view(bs, selnum, anncornum, ch)
        boxes_sml_new.append(box_sml_new)
        if len(masks_in_list) > 0:
            mask_sml_new = masks_in_list[0].masked_select(
                torch.zeros(bs, gridnum).to(masks_in_list[0].device).scatter(1, indices, 1).bool().unsqueeze(2).unsqueeze(
                    3).expand(bs, gridnum, anncornum, ch_m)).contiguous().view(bs, selnum, anncornum, ch_m)
            masks_sml_new.append(mask_sml_new)

        batchsize, dim, h, w = x_[0].size()
        i_new = x_[0].view(batchsize, dim, h * w).permute(0, 2, 1)
        bs, gridnum, ch = i_new.shape
        i_new = i_new.masked_select(
            torch.zeros(bs, gridnum).to(i_new.device).scatter(1, indices, 1).
                bool().unsqueeze(2).expand(bs, gridnum,ch)).contiguous().view(bs, selnum, ch)

        # Anchor-based Contrastive Learning
        x_new = self.linear_vs(i_new)
        y_new = self.linear_ts(y_['flat_lang_feat'].unsqueeze(1))
        if self.training:
            loss_out,loss_in = self.head(x_new, y_new,last_indices,wei = wei)
            return loss_out,loss_in
        else:
            predictions_s,maxval = self.head(x_new, y_new)
            predictions_list = [predictions_s]
            # box_pred = get_boxes(boxes_sml_new, predictions_list,self.class_num)
            if len(masks_in_list) > 0:
                box_pred, masks_in = get_masks(boxes_sml_new, masks_sml_new, predictions_list, self.class_num)
                mask_pred = batch_process_mask(protos, masks_in, box_pred[:,:,:4], (ih, iw), upsample=True)  # xywh
                mask_pred = mask_pred.squeeze(1)
                return box_pred, mask_pred,maxval
            else:
                mask_pred = None
                box_pred = get_boxes(boxes_sml_new, predictions_list,self.class_num)
                return box_pred, mask_pred,maxval




def get_boxes(boxes_sml, predictionslist,class_num):
    batchsize = predictionslist[0].size()[0]
    pred = []
    for i in range(len(predictionslist)):
        mask = predictionslist[i].squeeze(1)
        masked_pred = boxes_sml[i][mask]
        refined_pred = masked_pred.view(batchsize, -1, class_num+5)
        refined_pred[:, :, 0] = refined_pred[:, :, 0] - refined_pred[:, :, 2] / 2
        refined_pred[:, :, 1] = refined_pred[:, :, 1] - refined_pred[:, :, 3] / 2
        refined_pred[:, :, 2] = refined_pred[:, :, 0] + refined_pred[:, :, 2]
        refined_pred[:, :, 3] = refined_pred[:, :, 1] + refined_pred[:, :, 3]
        pred.append(refined_pred.data)
    boxes = torch.cat(pred, 1)
    score = boxes[:, :, 4]
    max_score, ind = torch.max(score, -1)
    ind_new = ind.unsqueeze(1).unsqueeze(1).repeat(1, 1, 5)
    box_new = torch.gather(boxes, 1, ind_new)
    return box_new


def get_masks(boxes_sml, masks_sml, predictionslist, class_num):
    batchsize = predictionslist[0].size()[0]
    pred = []
    pred_mask = []
    nm = 32
    for i in range(len(predictionslist)):
        mask = predictionslist[i].squeeze(1)
        masked_pred = boxes_sml[i][mask]
        masked_pred_mask = masks_sml[i][mask]
        refined_pred = masked_pred.view(batchsize, -1, class_num+5)
        refined_pred_mask = masked_pred_mask.view(batchsize, -1, nm)
        refined_pred[:, :, 0] = refined_pred[:, :, 0] - refined_pred[:, :, 2] / 2
        refined_pred[:, :, 1] = refined_pred[:, :, 1] - refined_pred[:, :, 3] / 2
        refined_pred[:, :, 2] = refined_pred[:, :, 0] + refined_pred[:, :, 2]
        refined_pred[:, :, 3] = refined_pred[:, :, 1] + refined_pred[:, :, 3]
        pred.append(refined_pred.data)
        pred_mask.append(refined_pred_mask.data)
    boxes = torch.cat(pred, 1)
    masks = torch.cat(pred_mask, 1)
    score = boxes[:, :, 4]
    max_score, ind = torch.max(score, -1)
    ind_new = ind.unsqueeze(1).unsqueeze(1).repeat(1, 1, 5)
    ind_new_mask = ind.unsqueeze(1).unsqueeze(1).repeat(1, 1, nm)
    box_new = torch.gather(boxes, 1, ind_new)
    mask_new = torch.gather(masks, 1, ind_new_mask)
    return box_new, mask_new
