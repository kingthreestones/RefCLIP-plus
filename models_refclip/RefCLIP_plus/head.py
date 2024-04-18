# coding=utf-8
import torch
import torch.nn as nn


def getContrast_in(vis_emb, lan_emb,gt=None,wei= None):
    sim_map = torch.einsum('bkd, byd -> byk', vis_emb, lan_emb)
    batchsize = sim_map.shape[0]

    # target = torch.full((batchsize,), sim_map.shape[2]-1).to(vis_emb.device)
    target = gt
    if wei==None:
        loss = nn.CrossEntropyLoss(reduction="mean")(sim_map.squeeze(1),target)
    else:
        loss = nn.CrossEntropyLoss(reduction="none")(sim_map.squeeze(1), target)
        loss = loss * wei
        loss = loss.mean()
    return loss


def getContrast(vis_emb, lan_emb):
    sim_map = torch.einsum('avd, bqd -> baqv',vis_emb,lan_emb)
    batchsize = sim_map.shape[0]
    max_sims,_ = sim_map.topk(k=2, dim=-1, largest=True, sorted=True)
    max_sims = max_sims.squeeze(2)

    # maxatt, _ = sim_map.max(dim=-1)  # [B1, B2, querys]: B1th sentence to B2th image
    # new_logits = torch.sum(maxatt, dim=-1).expand(maxatt.size(0), maxatt.size(1))


    # Negative Anchor Augmentation
    max_sim_0,max_sim_1 = max_sims[...,0],max_sims[...,1]
    max_sim_1 = max_sim_1.masked_select(~torch.eye(batchsize).bool().to(max_sim_1.device)).contiguous().view(batchsize,batchsize-1)
    new_logits = torch.cat([max_sim_0,max_sim_1],dim=1)

    target = torch.eye(batchsize).to(vis_emb.device)
    target_pred = torch.argmax(target, dim=1)
    loss = nn.CrossEntropyLoss(reduction="mean")(new_logits, target_pred)
    return loss

def getPrediction(vis_emb, lan_emb,gt=None):
    sim_map = torch.einsum('bkd, byd -> byk', vis_emb, lan_emb)
    maxval, v = sim_map.max(dim=2, keepdim=True)
    # v = gt.unsqueeze(1).unsqueeze(1)
    maxval = torch.sigmoid(maxval)
    predictions = torch.zeros_like(sim_map).to(sim_map.device).scatter(2,v.expand(sim_map.shape), 1).bool()
    return predictions,maxval

class WeakREChead(nn.Module):
    def __init__(self, __C):
        super(WeakREChead, self).__init__()
    def forward(self, vis_fs,lan_fs,gt_anc=None,wei=None):
        if self.training:
            loss_out = getContrast(vis_fs, lan_fs)
            if gt_anc is None:
                loss_in = torch.zeros_like(loss_out).to(loss_out.device)
            else:
                loss_in = getContrast_in(vis_fs, lan_fs,gt=gt_anc,wei=wei)
            return loss_out,loss_in
        else:
            predictions,maxval = getPrediction(vis_fs, lan_fs)
            return predictions,maxval










