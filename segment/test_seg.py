from utils_refclip.distributed import *
import torch.multiprocessing as mp
from utils_refclip.ckpt import *
from torch.nn.parallel import DistributedDataParallel as DDP
from utils_refclip.logging import *
import time
from utils_refclip import config
from datasets.dataloader_test import loader,RefCOCODataSet
from utils_refclip.utils import *

import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer)
from utils.segment.general import masks2segments, process_mask
from utils.torch_utils import select_device, smart_inference_mode



@smart_inference_mode()
def run(
    c_config = 'config/refcoco_test.yaml',
    weights=ROOT / 'yolov5s-seg.pt',  # model.pt path(s)
    source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / 'runs/predict-seg',  # save results to project/name
    name='exp',  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
    retina_masks=False,
):
    __C = config.load_cfg_from_cfg_file(c_config)
    gpu = __C.GPU
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


    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 2  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for ith_batch, data in enumerate(loader):
        ref_iter, image_iter, mask_iter, box_iter,gt_box_iter, mask_id, info_iter, im0 = data
        ref_iter = ref_iter.cuda( non_blocking=True)
        image_iter = image_iter.cuda( non_blocking=True)
        box_iter = box_iter.cuda( non_blocking=True)
        with dt[1]:
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred, proto = model(image_iter, augment=augment, visualize=visualize)[:2]
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
        
    
        # Process predictions
        for i, det in enumerate(pred):  # per image
            if bs > 1:
                im0 = im0[i].copy()
            else:
                im0 = im0.copy()
            seen += 1
            # if webcam:  # batch_size >= 1
            #     p, im0, frame = path[i], im0s[i].copy(), dataset.count
            #     s += f'{i}: '
            # else:
            #     p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem)  # im.txt
            s += '%gx%g ' % image_iter.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            masks = process_mask(proto[i], det[:, 6:], det[:, :4], image_iter.shape[2:], upsample=True)  # HWC
            det[:, :4] = scale_boxes(image_iter.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

            # Segments
            if save_txt:
                segments = [
                    scale_segments(im0.shape if retina_masks else image_iter.shape[2:], x, im0.shape, normalize=True)
                    for x in reversed(masks2segments(masks))]

            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Mask plotting
            annotator.masks(
                masks,
                colors=[colors(x, True) for x in det[:, 5]],
                im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() /
                255 if retina_masks else image_iter[i])

            # Write results
            for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                if save_txt:  # Write to file
                    seg = segments[j].reshape(-1)  # (n,2) to (n*2)
                    line = (cls, *seg, conf) if save_conf else (cls, *seg)  # label format
                    with open(f'{txt_path}.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    # annotator.draw.polygon(segments[j], outline=colors(c, True), width=3)
                if save_crop:
                    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

        # Stream results
        im0 = annotator.result()
        if view_img:
            if platform.system() == 'Linux' and p not in windows:
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            cv2.imshow(str(p), im0)
            if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                exit()

        # Save results (image with detections)
        if save_img:
            cv2.imwrite(save_path, im0)
        break

        # Print time (inference-only)
    LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c_config', type=str, default='config/refcoco_test.yaml')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-seg.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/predict-seg', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
