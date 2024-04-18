import torch

from utils_refclip.distributed import *
import torch.multiprocessing as mp
from utils_refclip.ckpt import *
from torch.nn.parallel import DistributedDataParallel as DDP
from utils_refclip.logging import *
import argparse
import time
from utils_refclip import config
from datasets.dataloader import loader, RefCOCODataSet
from tensorboardX import SummaryWriter
from utils_refclip.utils import *
from importlib import import_module
import torch.nn.functional as F
import torch.optim as Optim
from test import validate,validate_nomask
from utils_refclip.utils import EMA
import torch.nn as nn
from copy import deepcopy


def xyxy2xywhn(x, w=640, h=640):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right

    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


class ModelLoader:
    def __init__(self, __C):
        self.model_use = __C.MODEL
        model_moudle_path = 'models_refclip.' + self.model_use + '.net'
        self.model_moudle = import_module(model_moudle_path)

    def Net(self, __arg1, __arg2, __arg3):
        return self.model_moudle.Net(__arg1, __arg2, __arg3)


def train_one_epoch(__C,
                    __C2,
                    net,
                    net2,
                    optimizer,
                    optimizer2,
                    scheduler,
                    scheduler2,
                    loader,
                    scalar,
                    writer,
                    epoch,
                    rank,
                    ema=None,
                    ema2=None,
                    box_ap_base=None,
                    box_ap_stu=None,
                    ):
    net.train()
    net2.train()
    if __C.MULTIPROCESSING_DISTRIBUTED:
        loader.sampler.set_epoch(epoch)
    batches = len(loader)
    batch_time = AverageMeter('Time', ':6.5f')
    data_time = AverageMeter('Data', ':6.5f')
    losses = AverageMeter('Loss', ':.4f')
    losses_out = AverageMeter('Loss_out', ':.4f')
    losses_in = AverageMeter('Loss_in', ':.4f')
    losses_stu = AverageMeter('Loss_stu', ':.4f')
    lr = AverageMeter('lr', ':.5f')
    lr2 = AverageMeter('lr2', ':.5f')
    meters = [batch_time, data_time, losses, losses_out, losses_in, losses_stu, lr,lr2]
    meters_dict = {meter.name: meter for meter in meters}
    progress = ProgressMeter(__C.VERSION, __C.EPOCHS, len(loader), meters, prefix='Train: ')
    end = time.time()
    flag = 1
    flag1 = 1
    for ith_batch, data in enumerate(loader):
        data_time.update(time.time() - end)

        ref_iter,image_iter,mask_iter,box_iter,gt_box_iter,mask_id,info_iter,ref_iter2= data
        ref_iter = ref_iter.cuda(non_blocking=True)
        ref_iter2 = ref_iter2.cuda(non_blocking=True)
        image_iter = image_iter.cuda(non_blocking=True)
        box_iter = box_iter.cuda(non_blocking=True)
        mask_iter = mask_iter.cuda(non_blocking=True)
        image_iter2 = None
        if len(__C2.MULTI_SCALE) > 1:
            h, w = __C2.MULTI_SCALE[np.random.randint(0, len(__C2.MULTI_SCALE))]
            image_iter2 = F.interpolate(image_iter, (h, w))

        if scalar is not None:
            with th.cuda.amp.autocast():
                loss = net(image_iter, ref_iter)
        else:
            box_pred_stu = None
            maxsize_new = image_iter.size()[-1]
            if epoch >= __C.BURNIN:
                if epoch in __C.BACK:
                    net2.eval()
                    with torch.no_grad():
                        box_pred_stu_, _ = net2(image_iter, ref_iter2)

                    box_pred_stu = box_pred_stu_.clone().detach()
                    scores = box_pred_stu[:, 4]
                    indices = torch.where(scores > 0)


                    filtered_box_pred_stu = box_pred_stu[indices]

                    filtered_image_iter = image_iter[indices]
                    filtered_ref_iter = ref_iter[indices]
                    filtered_ref_iter2 = ref_iter2[indices]

                    wei = None
                    if filtered_box_pred_stu.shape[0] != 0:
                        box_pred_stu = filtered_box_pred_stu[:, :4]
                        box_pred_stu = xyxy2xywhn(box_pred_stu, w=maxsize_new, h=maxsize_new)
                        loss_out, loss_in = net(filtered_image_iter, filtered_ref_iter, box_pred_stu, wei=wei)
                        loss_stu = torch.zeros_like(loss_out).to(loss_out.device)
                        flag = 1
                        flag1 = 0

                    else:
                        if epoch in __C.BASESTOP:
                            flag1 = 1
                        else:
                            flag1 = 0
                            loss_out, loss_in = net(image_iter, ref_iter, box_pred_stu)
                        net.eval()

                        with torch.no_grad():
                            if ema is not None:
                                net_p = ema.get_apply_shadow()
                                net_p.eval()
                                box_pred_base_, mask_pred_base_,weight2 = net_p(image_iter, ref_iter)
                                del net_p
                            else:
                                box_pred_base_, mask_pred_base_,weight2 = net(image_iter, ref_iter)

                        net.train()
                        weight2 = weight2.squeeze(1).squeeze(1)
                        box_pred_base = box_pred_base_.clone().detach()
                        box_pred_base = box_pred_base.squeeze(1)[:, :4]
                        maxsize_old = image_iter.size()[-1]
                        maxsize_new = image_iter2.size()[-1]
                        ratio = maxsize_new / maxsize_old
                        box_pred_base = box_pred_base * ratio
                        box_pred_base = xyxy2xywhn(box_pred_base, w=maxsize_new, h=maxsize_new)
                        box_pred_base = box_pred_base.unsqueeze(1)
                        net2.train()
                        loss_stu, loss_det, loss_seg = net2(image_iter2, ref_iter2, det_label=box_pred_base,
                                                            seg_label=None, weight=weight2)
                        flag = 0
                        if epoch in __C.BASESTOP:
                            loss_out = torch.zeros_like(loss_stu).to(loss_stu.device)
                            loss_in = torch.zeros_like(loss_stu).to(loss_stu.device)

                    loss_out = 0 * loss_out
                else:

                    if epoch in __C.BASESTOP:
                        flag1 = 1
                    else:
                        flag1 = 0
                        loss_out, loss_in = net(image_iter, ref_iter, box_pred_stu)
                    net.eval()

                    with torch.no_grad():
                        if ema is not None:
                            net_p = ema.get_apply_shadow()
                            net_p.eval()
                            box_pred_base_, mask_pred_base_,weight2 = net_p(image_iter, ref_iter)
                            del net_p
                        else:
                            box_pred_base_, mask_pred_base_,weight2 = net(image_iter, ref_iter)

                    net.train()
                    weight2 = weight2.squeeze(1).squeeze(1)
                    box_pred_base = box_pred_base_.clone().detach()
                    box_pred_base = box_pred_base.squeeze(1)[:, :4]
                    indices = torch.where(weight2 > 0)
                    # 根据索引取出子集
                    filtered_box_pred_tea = box_pred_base[indices]
                    if filtered_box_pred_tea.shape[0] != 0:
                        box_pred_base = filtered_box_pred_tea
                        image_iter = image_iter[indices]
                        image_iter2 = image_iter2[indices]
                        ref_iter = ref_iter[indices]
                        ref_iter2 = ref_iter2[indices]
                        weight2 = weight2[indices]

                    maxsize_old = image_iter.size()[-1]
                    maxsize_new = image_iter2.size()[-1]
                    ratio = maxsize_new / maxsize_old
                    box_pred_base = box_pred_base * ratio
                    box_pred_base = xyxy2xywhn(box_pred_base, w=maxsize_new, h=maxsize_new)
                    box_pred_base = box_pred_base.unsqueeze(1)
                    net2.train()
                    if epoch in __C.BASENOWEIGHT:
                        weight2 = torch.ones_like(weight2).to(weight2.device)
                    loss_stu, loss_det, loss_seg = net2(image_iter2, ref_iter2, det_label=box_pred_base,
                                                        seg_label=None, weight=weight2)
                    flag = 0
                    if epoch in __C.BASESTOP:
                        loss_out = torch.zeros_like(loss_stu).to(loss_stu.device)
                        loss_in = torch.zeros_like(loss_stu).to(loss_stu.device)


            else:
                loss_out, loss_in = net(image_iter, ref_iter, box_pred_stu)
                loss_stu = torch.zeros_like(loss_out).to(loss_out.device)
                flag1 = 0

            loss = loss_out + loss_in + loss_stu

        if flag1 == 0:
            optimizer.zero_grad()
        if flag == 0:
            optimizer2.zero_grad()
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        if scalar is not None:
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            if __C.GRAD_NORM_CLIP > 0:
                nn.utils.clip_grad_norm_(
                    net.parameters(),
                    __C.GRAD_NORM_CLIP
                )
            scalar.update()
        else:
            loss.backward()
            if __C.GRAD_NORM_CLIP > 0:
                nn.utils.clip_grad_norm_(
                    net.parameters(),
                    __C.GRAD_NORM_CLIP
                )
                nn.utils.clip_grad_norm_(
                    net2.parameters(),
                    __C.GRAD_NORM_CLIP
                )
            if flag1 == 0:
                optimizer.step()
            if flag == 0:
                optimizer2.step()
        if flag1 == 0:
            scheduler.step()
        if flag == 0:
            scheduler2.step()
        if flag1 == 0:
            if ema is not None:
                ema.update_params()
        if flag == 0:
            if ema2 is not None:
                ema2.update_params()
        losses.update(loss.item(), image_iter.size(0))
        losses_out.update(loss_out.item(), image_iter.size(0))
        losses_in.update(loss_in.item(), image_iter.size(0))
        losses_stu.update(loss_stu.item(), image_iter.size(0))
        lr.update(optimizer.param_groups[0]["lr"], -1)
        lr2.update(optimizer2.param_groups[0]["lr"], -1)

        reduce_meters(meters_dict, rank, __C)
        if main_process(__C, rank):
            global_step = epoch * batches + ith_batch
            writer.add_scalar("loss/train", losses.avg_reduce, global_step=global_step)
            writer.add_scalar("loss_out/train", losses_out.avg_reduce, global_step=global_step)
            writer.add_scalar("loss_in/train", losses_in.avg_reduce, global_step=global_step)
            writer.add_scalar("loss_stu/train", losses_stu.avg_reduce, global_step=global_step)
            if ith_batch % __C.PRINT_FREQ == 0 or ith_batch == len(loader):
                progress.display(epoch, ith_batch)
        # break
        batch_time.update(time.time() - end)
        end = time.time()


def main_worker(gpu, __C,__C2):
    global best_det_acc,best_det_acc2,best_seg_acc
    best_det_acc = 0.
    best_det_acc2 = 0.
    best_seg_acc = 0.
    if __C.MULTIPROCESSING_DISTRIBUTED:
        if __C.DIST_URL == "env://" and __C.RANK == -1:
            __C.RANK = int(os.environ["RANK"])
        if __C.MULTIPROCESSING_DISTRIBUTED:
            __C.RANK = __C.RANK * len(__C.GPU) + gpu
        dist.init_process_group(backend=dist.Backend('NCCL'), init_method=__C.DIST_URL, world_size=__C.WORLD_SIZE,
                                rank=__C.RANK)

    train_set = RefCOCODataSet(__C, split='train')
    train_loader = loader(__C, train_set, gpu, shuffle=(not __C.MULTIPROCESSING_DISTRIBUTED), drop_last=True)

    val_set = RefCOCODataSet(__C, split='val')
    val_loader = loader(__C, val_set, gpu, shuffle=False)

    net = ModelLoader(__C).Net(
        __C,
        train_set.pretrained_emb,
        train_set.token_size
    )
    net2 = ModelLoader(__C2).Net(
        __C2,
        train_set.pretrained_emb,
        train_set.token_size
    )
    # optimizer
    params = filter(lambda p: p.requires_grad, net.parameters())  # split_weights(net)
    std_optim = getattr(Optim, __C.OPT)

    eval_str = 'params, lr=%f' % __C.LR
    for key in __C.OPT_PARAMS:
        eval_str += ' ,' + key + '=' + str(__C.OPT_PARAMS[key])
    optimizer = eval('std_optim' + '(' + eval_str + ')')

    # optimizer2
    params2 = filter(lambda p: p.requires_grad, net2.parameters())  # split_weights(net)
    std_optim2 = getattr(Optim, __C.OPT)

    eval_str2 = 'params2, lr=%f' % __C.LR
    for key in __C.OPT_PARAMS:
        eval_str2 += ' ,' + key + '=' + str(__C.OPT_PARAMS[key])
    optimizer2 = eval('std_optim2' + '(' + eval_str2 + ')')

    ema = None
    ema2 = None

    if __C.MULTIPROCESSING_DISTRIBUTED:
        torch.cuda.set_device(gpu)
        net = DDP(net.cuda(), device_ids=[gpu], find_unused_parameters=True,broadcast_buffers=False)
        net2 = DDP(net2.cuda(), device_ids=[gpu], find_unused_parameters=True,broadcast_buffers=False)
    elif len(gpu) == 1:
        net.cuda()
        net2.cuda()
    else:
        net = DP(net.cuda())

    if main_process(__C, gpu):
        print(__C)
        # print(net)
        total = sum([param.nelement() for param in net.parameters()])
        print('  + Number of all params: %.2fM' % (total / 1e6))  # 每一百万为一个单位
        total = sum([param.nelement() for param in net.parameters() if param.requires_grad])
        print('  + Number of trainable params: %.2fM' % (total / 1e6))  # 每一百万为一个单位

    scheduler = get_lr_scheduler(__C, optimizer, len(train_loader))
    scheduler2 = get_lr_scheduler(__C2, optimizer2, len(train_loader))

    start_epoch = 0

    if os.path.isfile(__C.RESUME_PATH):
        checkpoint = torch.load(__C.RESUME_PATH, map_location=lambda storage, loc: storage.cuda())
        new_dict = {}
        for k in checkpoint['state_dict']:
            if 'module.' in k:
                new_k = k.replace('module.', '')
                new_dict[new_k] = checkpoint['state_dict'][k]
        if len(new_dict.keys()) == 0:
            new_dict = checkpoint['state_dict']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        if main_process(__C, gpu):
            print("==> loaded checkpoint from {}\n".format(__C.RESUME_PATH) +
                  "==> epoch: {} lr: {} ".format(checkpoint['epoch'], checkpoint['lr']))

    if __C.AMP:
        assert th.__version__ >= '1.6.0', \
            "Automatic Mixed Precision training only supported in PyTorch-1.6.0 or higher"
        scalar = th.cuda.amp.GradScaler()
    else:
        scalar = None

    if main_process(__C, gpu):
        writer = SummaryWriter(log_dir=os.path.join(__C.LOG_PATH, str(__C.VERSION)))
    else:
        writer = None

    save_ids = np.random.randint(1, len(val_loader) * __C.BATCH_SIZE, 100) if __C.LOG_IMAGE else None
    box_ap, box_ap2 ,mask_ap= 0., 0.,0.
    for ith_epoch in range(start_epoch, __C.EPOCHS):
        if __C.USE_EMA and ema is None:
            ema = EMA(net, 0.9997)
            ema2 = EMA(net2, 0.9997)
        train_one_epoch(__C,__C2, net,net2, optimizer,optimizer2, scheduler,scheduler2, train_loader, scalar, writer,
                        ith_epoch, gpu, ema,ema2,box_ap_base=box_ap,box_ap_stu=box_ap2)
        box_ap,mask_ap,box_ap2 = validate(__C,__C2, net,net2, val_loader, writer, ith_epoch, gpu, val_set.ix_to_token, save_ids=save_ids,
                                   ema=ema,ema2=ema2)
        if main_process(__C, gpu):
            if ema is not None:
                ema.apply_shadow()
            if ema2 is not None:
                ema2.apply_shadow()
            if box_ap > best_det_acc:
                best_det_acc = box_ap
                torch.save({'epoch': ith_epoch + 1, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(), 'lr': optimizer.param_groups[0]["lr"], },
                           os.path.join(__C.LOG_PATH, str(__C.VERSION), 'ckpt', 'det_best_tea.pth'))
            if box_ap2 > best_det_acc2:
                best_det_acc2 = box_ap2
                torch.save({'epoch': ith_epoch + 1, 'state_dict': net2.state_dict(), 'optimizer': optimizer2.state_dict(),
                            'scheduler': scheduler2.state_dict(), 'lr': optimizer2.param_groups[0]["lr"], },
                           os.path.join(__C.LOG_PATH, str(__C.VERSION), 'ckpt', 'det_best_stu.pth'))
            if mask_ap>best_seg_acc:
                best_seg_acc=mask_ap
                torch.save({'epoch': ith_epoch + 1, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),'lr':optimizer.param_groups[0]["lr"],},
                           os.path.join(__C.LOG_PATH, str(__C.VERSION),'ckpt', 'seg_best.pth'))
                
            if ema is not None:
                ema.restore()
            if ema2 is not None:
                ema2.restore()
    if __C.MULTIPROCESSING_DISTRIBUTED:
        cleanup_distributed()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, default='./config/refcoco_base.yaml')
    parser.add_argument('--config2', type=str, required=True, default='./config/simrec.yaml')
    args = parser.parse_args()
    assert args.config is not None
    __C = config.load_cfg_from_cfg_file(args.config)
    __C2 = config.load_cfg_from_cfg_file(args.config2)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in __C.GPU)
    setup_unique_version(__C)
    seed_everything(__C.SEED)
    N_GPU = len(__C.GPU)

    if not os.path.exists(os.path.join(__C.LOG_PATH, str(__C.VERSION))):
        os.makedirs(os.path.join(__C.LOG_PATH, str(__C.VERSION), 'ckpt'), exist_ok=True)

    if N_GPU == 1:
        __C.MULTIPROCESSING_DISTRIBUTED = False
    else:
        # turn on single or multi node multi gpus training
        __C.MULTIPROCESSING_DISTRIBUTED = True
        __C.WORLD_SIZE *= N_GPU
        __C.DIST_URL = f"tcp://127.0.0.1:{find_free_port()}"
    if __C.MULTIPROCESSING_DISTRIBUTED:
        mp.spawn(main_worker, args=(__C,__C2), nprocs=N_GPU, join=True)
    else:
        main_worker(__C.GPU, __C,__C2)


if __name__ == '__main__':
    main()
