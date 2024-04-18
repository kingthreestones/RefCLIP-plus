from utils_refclip.distributed import *
import torch.multiprocessing as mp
from utils_refclip.ckpt import *
from torch.nn.parallel import DistributedDataParallel as DDP
from utils_refclip.logging import *
import argparse
import time
from utils_refclip import config
from datasets.dataloader import loader,RefCOCODataSet
from tensorboardX import SummaryWriter
from utils_refclip.utils import *
import torch.optim as Optim
from importlib import import_module
import cv2
def normed2original(image,mean=None,std=None,transpose=True):
    """
    :param image: 3,h,w
    :param mean: 3
    :param std: 3
    :return:
    """
    if std is not None:
        std=torch.from_numpy(np.array(std)).to(image.device).float()
        image=image*std.unsqueeze(-1).unsqueeze(-1)
    if mean is not None:
        mean=torch.from_numpy(np.array(mean)).to(image.device).float()
        image=image+mean.unsqueeze(-1).unsqueeze(-1)
    if transpose:
        image=image.permute(1,2,0)
    return image.cpu().numpy()
def draw_visualization(image,sent,pred_box,gt_box,draw_text=True,savepath=None):
    # image=(image*255).astype(np.uint8)
    image=np.ascontiguousarray(image)
    left, top, right, bottom,_ = (pred_box).astype('int32')
    gt_left, gt_top, gt_right, gt_bottom = (gt_box).astype('int32')
    colors=[(255,0,0),(0,255,0),(0,191,255)]

    cv2.rectangle(image, (left, top ), (right , bottom ), colors[0], 2)
    cv2.rectangle(image, (gt_left, gt_top), (gt_right, gt_bottom), colors[1], 2)



    if draw_text:
        cv2.putText(image,
                    '{:%.2f}' % pred_box[-1],
                    (left, max(top - 3, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, colors[0], 2)
        cv2.putText(image,
                    'ground_truth',
                    (gt_left, max(gt_top - 3, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, colors[1], 2)
        cv2.putText(image,
                    str(sent),
                    (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, colors[2], 2)
    return image


class ModelLoader:
    def __init__(self, __C):

        self.model_use = __C.MODEL
        model_moudle_path = 'models_refclip.' + self.model_use + '.net'
        self.model_moudle = import_module(model_moudle_path)

    def Net(self, __arg1, __arg2, __arg3):
        return self.model_moudle.Net(__arg1, __arg2, __arg3)


def validate_nomask(__C,
             __C2,
             net,
             net2,
             loader,
             writer,
             epoch,
             rank,
             ix_to_token,
             save_ids=None,
             prefix='Val',
             ema=None,
             ema2=None):
    if ema is not None:
        ema.apply_shadow()
    if ema2 is not None:
        ema2.apply_shadow()
    net.eval()
    net2.eval()

    batches = len(loader)
    batch_time = AverageMeter('Time', ':6.5f')
    data_time = AverageMeter('Data', ':6.5f')
    losses = AverageMeter('Loss', ':.4f')
    box_ap = AverageMeter('BoxIoU@0.5', ':6.2f')
    box_ap2 = AverageMeter('BoxIoU2@0.5', ':6.2f')
    
    meters = [batch_time, data_time, losses, box_ap,box_ap2]
    meters_dict = {meter.name: meter for meter in meters}
    progress = ProgressMeter(__C.VERSION, __C.EPOCHS, len(loader), meters, prefix=prefix+': ')
    with torch.no_grad():
        end = time.time()
        for ith_batch, data in enumerate(loader):
            ref_iter, image_iter, mask_iter, box_iter,gt_box_iter, mask_id, info_iter, ref_iter2 = data
            ref_iter = ref_iter.cuda( non_blocking=True)
            ref_iter2 = ref_iter2.cuda(non_blocking=True)
            image_iter = image_iter.cuda( non_blocking=True)
            box_iter = box_iter.cuda( non_blocking=True)
            box,mask,_= net(image_iter, ref_iter)
            box2,_ = net2(image_iter, ref_iter2)


            gt_box_iter=gt_box_iter.squeeze(1)
            gt_box_iter[:, 2] = (gt_box_iter[:, 0] + gt_box_iter[:, 2])
            gt_box_iter[:, 3] = (gt_box_iter[:, 1] + gt_box_iter[:, 3])
            gt_box_iter=gt_box_iter.cpu().numpy()
            info_iter=info_iter.cpu().numpy()
            box = box.squeeze(1).cpu().numpy()
            box2 = box2.squeeze(1).cpu().numpy()
            pred_box_vis=box.copy()

            #predictions to gt
            for i in range(len(gt_box_iter)):
                box[i] = yolobox2label(box[i],info_iter[i])
                box2[i] = yolobox2label(box2[i], info_iter[i])
            box_iou=batch_box_iou(torch.from_numpy(gt_box_iter),torch.from_numpy(box)).cpu().numpy()
            box_iter = box_iter.view(box_iter.shape[0], -1) * __C.INPUT_SHAPE[0]
            box_iter[:, 0] = box_iter[:, 0] - 0.5 * box_iter[:, 2]
            box_iter[:, 1] = box_iter[:, 1] - 0.5 * box_iter[:, 3]
            box_iter[:, 2] = box_iter[:, 0] + box_iter[:, 2]
            box_iter[:, 3] = box_iter[:, 1] + box_iter[:, 3]

            for i, box_pred in enumerate(pred_box_vis):
                if writer is not None and save_ids is not None and ith_batch * __C.BATCH_SIZE + i in save_ids:
                    ixs = ref_iter[i].cpu().numpy()
                    words = []
                    for ix in ixs:
                        if ix > 0:
                            words.append(ix_to_token[ix])
                    sent = ' '.join(words)

                    # det_image=draw_visualization(normed2original(image_iter[i],__C.MEAN,__C.STD),sent,pred_box_vis[i].cpu().numpy(),box_iter[i].cpu().numpy())
                    det_image = draw_visualization(normed2original(image_iter[i], __C.MEAN, __C.STD), sent,
                                                   pred_box_vis[i], box_iter[i].cpu().numpy())
                    writer.add_image('image/' + prefix + '_' + str(ith_batch * __C.BATCH_SIZE + i) + '_det', det_image,
                                     epoch, dataformats='HWC')

            box_ap.update((box_iou>0.5).astype(np.float32).mean()*100., box_iou.shape[0])
            box_iou2 = batch_box_iou(torch.from_numpy(gt_box_iter), torch.from_numpy(box2)).cpu().numpy()
            box_ap2.update((box_iou2 > 0.5).astype(np.float32).mean() * 100., box_iou2.shape[0])
            reduce_meters(meters_dict, rank, __C)
            if (ith_batch % __C.PRINT_FREQ == 0 or ith_batch==(len(loader)-1)) and main_process(__C,rank):
                progress.display(epoch, ith_batch)
            batch_time.update(time.time() - end)
            end = time.time()

        if main_process(__C,rank) and writer is not None:
            writer.add_scalar("Acc/BoxIoU@0.5", box_ap.avg_reduce, global_step=epoch)
            writer.add_scalar("Acc/BoxIoU2@0.5", box_ap2.avg_reduce, global_step=epoch)
    if ema is not None:
        ema.restore()
    if ema2 is not None:
        ema2.restore()
    return box_ap.avg_reduce,0.,box_ap2.avg_reduce





def validate(__C,
             __C2,
             net,
             net2,
             loader,
             writer,
             epoch,
             rank,
             ix_to_token,
             save_ids=None,
             prefix='Val',
             ema=None,
             ema2=None):
    if ema is not None:
        ema.apply_shadow()
    if ema2 is not None:
        ema2.apply_shadow()
    net.eval()
    net2.eval()

    batches = len(loader)
    batch_time = AverageMeter('Time', ':6.5f')
    data_time = AverageMeter('Data', ':6.5f')
    losses = AverageMeter('Loss', ':.4f')
    box_ap = AverageMeter('BoxIoU@0.5', ':6.2f')
    box_ap2 = AverageMeter('BoxIoU2@0.5', ':6.2f')
    mask_ap = AverageMeter('MaskIoU', ':6.2f')
    inconsistency_error = AverageMeter('IE', ':6.2f')
    mask_aps={}
    for item in np.arange(0.5, 1, 0.05):
        mask_aps[item]=[]
    meters = [batch_time, data_time, losses, box_ap,box_ap2, mask_ap, inconsistency_error]
    meters_dict = {meter.name: meter for meter in meters}
    progress = ProgressMeter(__C.VERSION, __C.EPOCHS, len(loader), meters, prefix=prefix+': ')
    with torch.no_grad():
        end = time.time()
        for ith_batch, data in enumerate(loader):
            ref_iter, image_iter, mask_iter, box_iter,gt_box_iter, mask_id, info_iter, ref_iter2 = data
            ref_iter = ref_iter.cuda( non_blocking=True)
            ref_iter2 = ref_iter2.cuda(non_blocking=True)
            image_iter = image_iter.cuda( non_blocking=True)
            box_iter = box_iter.cuda( non_blocking=True)
            box,mask,_= net(image_iter, ref_iter)
            box2,_ = net2(image_iter, ref_iter2)


            gt_box_iter=gt_box_iter.squeeze(1)
            gt_box_iter[:, 2] = (gt_box_iter[:, 0] + gt_box_iter[:, 2])
            gt_box_iter[:, 3] = (gt_box_iter[:, 1] + gt_box_iter[:, 3])
            gt_box_iter=gt_box_iter.cpu().numpy()
            info_iter=info_iter.cpu().numpy()
            box = box.squeeze(1).cpu().numpy()
            box2 = box2.squeeze(1).cpu().numpy()
            pred_box_vis=box2.copy()

            #predictions to gt
            for i in range(len(gt_box_iter)):
                box[i] = yolobox2label(box[i],info_iter[i])
                box2[i] = yolobox2label(box2[i], info_iter[i])
            box_iou=batch_box_iou(torch.from_numpy(gt_box_iter),torch.from_numpy(box)).cpu().numpy()
            seg_iou=[]
            mask=mask.cpu().numpy()
            box_iter = box_iter.view(box_iter.shape[0], -1) * __C.INPUT_SHAPE[0]
            box_iter[:, 0] = box_iter[:, 0] - 0.5 * box_iter[:, 2]
            box_iter[:, 1] = box_iter[:, 1] - 0.5 * box_iter[:, 3]
            box_iter[:, 2] = box_iter[:, 0] + box_iter[:, 2]
            box_iter[:, 3] = box_iter[:, 1] + box_iter[:, 3]

            for i, mask_pred in enumerate(mask):
                if writer is not None and save_ids is not None and ith_batch * __C.BATCH_SIZE + i in save_ids:
                    ixs = ref_iter[i].cpu().numpy()
                    words = []
                    for ix in ixs:
                        if ix > 0:
                            words.append(ix_to_token[ix])
                    sent = ' '.join(words)

                    # det_image=draw_visualization(normed2original(image_iter[i],__C.MEAN,__C.STD),sent,pred_box_vis[i].cpu().numpy(),box_iter[i].cpu().numpy())
                    det_image=draw_visualization(normed2original(image_iter[i],__C.MEAN,__C.STD),sent,pred_box_vis[i],box_iter[i].cpu().numpy())
                    writer.add_image('image/' + prefix + '_' + str(ith_batch * __C.BATCH_SIZE + i) + '_det', det_image,epoch, dataformats='HWC')
                    writer.add_image('image/' + prefix + '_' + str(ith_batch * __C.BATCH_SIZE + i) + '_seg', (mask[i,None]*255).astype(np.uint8))
                mask_gt=np.load(os.path.join(__C.MASK_PATH[__C.DATASET],'%d.npy'%mask_id[i]))

                mask_pred=mask_processing(mask_pred,info_iter[i])

                # view gt masks and pred masks
                # writer.add_image('image/' + str(ith_batch * __C.BATCH_SIZE + i) + '_gt-seg', (mask_gt[None]*255).astype(np.uint8))
                # writer.add_image('image/' + str(ith_batch * __C.BATCH_SIZE + i) + '_pred-seg', (mask_pred[None]*255).astype(np.uint8))


                single_seg_iou,single_seg_ap=mask_iou(mask_gt,mask_pred)
                for item in np.arange(0.5, 1, 0.05):
                    mask_aps[item].append(single_seg_ap[item]*100.)
                seg_iou.append(single_seg_iou)
            seg_iou=np.array(seg_iou).astype(np.float32)

            ie=(box_iou>=0.5).astype(np.float32)*(seg_iou<0.5).astype(np.float32)+(box_iou<0.5).astype(np.float32)*(seg_iou>=0.5).astype(np.float32)
            inconsistency_error.update(ie.mean()*100., ie.shape[0])
            box_ap.update((box_iou>0.5).astype(np.float32).mean()*100., box_iou.shape[0])
            box_iou2 = batch_box_iou(torch.from_numpy(gt_box_iter), torch.from_numpy(box2)).cpu().numpy()
            box_ap2.update((box_iou2 > 0.5).astype(np.float32).mean() * 100., box_iou2.shape[0])
            mask_ap.update(seg_iou.mean()*100., seg_iou.shape[0])
            reduce_meters(meters_dict, rank, __C)
            if (ith_batch % __C.PRINT_FREQ == 0 or ith_batch==(len(loader)-1)) and main_process(__C,rank):
                progress.display(epoch, ith_batch)
            batch_time.update(time.time() - end)
            end = time.time()

        if main_process(__C,rank) and writer is not None:
            writer.add_scalar("Acc/BoxIoU@0.5", box_ap.avg_reduce, global_step=epoch)
            writer.add_scalar("Acc/MaskIoU", mask_ap.avg_reduce, global_step=epoch)
            writer.add_scalar("Acc/BoxIoU2@0.5", box_ap2.avg_reduce, global_step=epoch)
            writer.add_scalar("Acc/IE", inconsistency_error.avg_reduce, global_step=epoch)
            for item in mask_aps:
                writer.add_scalar("Acc/MaskIoU@%.2f"%item, np.array(mask_aps[item]).mean(), global_step=epoch)
    if ema is not None:
        ema.restore()
    if ema2 is not None:
        ema2.restore()
    return box_ap.avg_reduce,mask_ap.avg_reduce,box_ap2.avg_reduce


def main_worker(gpu, __C,__C2):
    global best_det_acc,best_det_acc2,best_mask_acc
    best_det_acc = 0.
    best_mask_acc = 0.
    best_det_acc2 = 0.
    if __C.MULTIPROCESSING_DISTRIBUTED:
        if __C.DIST_URL == "env://" and __C.RANK == -1:
            __C.RANK = int(os.environ["RANK"])
        if __C.MULTIPROCESSING_DISTRIBUTED:
            __C.RANK = __C.RANK* len(__C.GPU) + gpu
        dist.init_process_group(backend=dist.Backend('NCCL'), init_method=__C.DIST_URL, world_size=__C.WORLD_SIZE, rank=__C.RANK)

    train_set=RefCOCODataSet(__C,split='train')
    train_loader=loader(__C,train_set,gpu,shuffle=(not __C.MULTIPROCESSING_DISTRIBUTED))

    loaders=[]
    prefixs=['val']
    val_set=RefCOCODataSet(__C,split='val')
    val_loader=loader(__C,val_set,gpu,shuffle=False)
    loaders.append(val_loader)
    if __C.DATASET=='refcoco' or __C.DATASET=='refcoco+':
        testA=RefCOCODataSet(__C,split='testA')
        testA_loader=loader(__C,testA,gpu,shuffle=False)
        testB=RefCOCODataSet(__C,split='testB')
        testB_loader=loader(__C,testB,gpu,shuffle=False)
        prefixs.extend(['testA','testB'])
        loaders.extend([testA_loader,testB_loader])
    elif __C.DATASET=='referit':
        test=RefCOCODataSet(__C,split='test')
        test_loader=loader(__C,test,gpu,shuffle=False)
        prefixs.append('test')
        loaders.append(test_loader)

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

    #optimizer
    std_optim = getattr(Optim, __C.OPT)
    params = filter(lambda p: p.requires_grad, net.parameters())  # split_weights(net)
    eval_str = 'params, lr=%f'%__C.LR
    for key in __C.OPT_PARAMS:
        eval_str += ' ,' + key + '=' + str(__C.OPT_PARAMS[key])
    optimizer=eval('std_optim' + '(' + eval_str + ')')


    if __C.MULTIPROCESSING_DISTRIBUTED:
        torch.cuda.set_device(gpu)
        net = DDP(net.cuda(), device_ids=[gpu],find_unused_parameters=True)
    elif len(gpu)==1:
        net.cuda()
        net2.cuda()
    else:
        net = DP(net.cuda())
    if main_process(__C, gpu):
        print(__C)
        total = sum([param.nelement() for param in net.parameters()])
        print('  + Number of all params: %.2fM' % (total / 1e6))  # 每一百万为一个单位
        total = sum([param.nelement() for param in net.parameters() if param.requires_grad])
        print('  + Number of trainable params: %.2fM' % (total / 1e6))  # 每一百万为一个单位


    if os.path.isfile(__C.RESUME_PATH):
        checkpoint = torch.load(__C.RESUME_PATH,map_location=lambda storage, loc: storage.cuda() )
        new_dict = {}
        for k in checkpoint['state_dict']:
            if 'module.' in k:
                new_k = k.replace('module.', '')
                new_dict[new_k] = checkpoint['state_dict'][k]
        if len(new_dict.keys()) == 0:
            new_dict = checkpoint['state_dict']
        
        net.load_state_dict(new_dict)

        if main_process(__C,gpu):
            print("==> loaded checkpoint from {}\n".format(__C.RESUME_PATH) +
                  "==> epoch: {} lr: {} ".format(checkpoint['epoch'],checkpoint['lr']))
    if os.path.isfile(__C2.RESUME_PATH):
        checkpoint = torch.load(__C2.RESUME_PATH,map_location=lambda storage, loc: storage.cuda() )
        new_dict = {}
        for k in checkpoint['state_dict']:
            if 'module.' in k:
                new_k = k.replace('module.', '')
                new_dict[new_k] = checkpoint['state_dict'][k]
        if len(new_dict.keys()) == 0:
            new_dict = checkpoint['state_dict']
        net2.load_state_dict(new_dict)

        if main_process(__C,gpu):
            print("==> loaded checkpoint from {}\n".format(__C2.RESUME_PATH) +
                  "==> epoch: {} lr: {} ".format(checkpoint['epoch'],checkpoint['lr']))
    if __C.AMP:
        assert th.__version__ >= '1.6.0', \
            "Automatic Mixed Precision training only supported in PyTorch-1.6.0 or higher"
        scalar = th.cuda.amp.GradScaler()
    else:
        scalar = None

    if main_process(__C,gpu):
        writer = SummaryWriter(log_dir=os.path.join(__C.LOG_PATH,str(__C.VERSION)))
    else:
        writer = None

    save_ids=np.random.randint(1, len(val_loader) * __C.BATCH_SIZE, 100) if __C.LOG_IMAGE else None
    for loader_,prefix_ in zip(loaders,prefixs):
        box_ap,mask_ap,box_ap2 = validate_nomask(__C,__C2, net,net2, loader_, writer, 0, gpu, val_set.ix_to_token, save_ids=save_ids,prefix=prefix_)
        print(box_ap,mask_ap,box_ap2)


def main():
    parser = argparse.ArgumentParser(description="RefCLIP")
    parser.add_argument('--config', type=str, default='config/refcoco.yaml')
    parser.add_argument('--config2', type=str, default='./config/simrec.yaml')
    parser.add_argument('--eval-weights', type=str, default='')
    parser.add_argument('--eval-weights2', type=str, default='')
    args=parser.parse_args()
    assert args.config is not None
    __C = config.load_cfg_from_cfg_file(args.config)
    __C2 = config.load_cfg_from_cfg_file(args.config2)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in __C.GPU)
    setup_unique_version(__C)
    seed_everything(__C.SEED)
    N_GPU=len(__C.GPU)
    __C.RESUME_PATH=args.eval_weights
    __C2.RESUME_PATH=args.eval_weights2
    if not os.path.exists(os.path.join(__C.LOG_PATH,str(__C.VERSION))):
        os.makedirs(os.path.join(__C.LOG_PATH,str(__C.VERSION),'ckpt'),exist_ok=True)

    if N_GPU == 1:
        __C.MULTIPROCESSING_DISTRIBUTED = False
    else:
        # turn on single or multi node multi gpus training
        __C.MULTIPROCESSING_DISTRIBUTED = True
        __C.WORLD_SIZE *= N_GPU
        __C.DIST_URL = f"tcp://127.0.0.1:{find_free_port()}"
    if __C.MULTIPROCESSING_DISTRIBUTED:
        mp.spawn(main_worker, args=(__C,), nprocs=N_GPU, join=True)
    else:
        main_worker(__C.GPU,__C,__C2)


if __name__ == '__main__':
    main()
