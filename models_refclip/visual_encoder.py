# coding=utf-8

from models.common import DetectMultiBackend
from utils.torch_utils import select_device

backbone_dict={
    'yolov5':DetectMultiBackend,
    'yolov3':DetectMultiBackend
}
def visual_encoder(__C):
    weights = __C.PRETRAIN_WEIGHT
    device = select_device(__C.GPU[0])
    dnn = False
    data =  "./data/vg.yaml" # 'coco128.yaml'
    half = False
    print(weights)
    vis_enc=backbone_dict[__C.VIS_ENC](weights, device=device, dnn=dnn, data=data, fp16=half)
    return vis_enc

