import os
import time
import shutil
from turtle import update
import cv2
import numpy as np
from PIL import Image
from mindspore import nn
from mindspore.ops import operations as ops
from mindspore.train.callback import Callback
from mindspore.common.tensor import Tensor
import matplotlib.pyplot as plt
from mindspore import load_checkpoint, load_param_into_net
from src.transunet import TransUNet
from src.unet import UNet
from config import cfg
import mindspore as msp
import mindspore.numpy as numpy
from tqdm import tqdm
from mydataset import DatasetGenerator
import mindspore.dataset as ds
from mindspore import context
import logging
# from src.unet_in_unet import Unet_in_Unet


def get_iou(label, predict):
    
    image_mask = label.asnumpy()
    predict = predict.asnumpy()
    interArea = np.multiply(predict, image_mask)
    tem = predict + image_mask
    unionArea = tem - interArea
    inter = np.sum(interArea)
    union = np.sum(unionArea) + 0.000000001
    iou_tem = inter / union

    return iou_tem

def get_dice(label, predict):


    image_mask = label.asnumpy()
    predict = predict.asnumpy()
    intersection = (predict*image_mask).sum()
    dice = (2. *intersection) /(predict.sum()+image_mask.sum())
    return dice


from sklearn.metrics import precision_recall_curve
def get_pr(label, predict):
    # For each class
    precision = dict()
    recall = dict()
    # n_classes = label.shape[0]
    C, H, W = label.shape
    reshape = ops.Reshape()
    # label = reshape(label.transpose((1,2,0)),(-1, 1)).asnumpy()
    # predict = reshape(predict.transpose((1,2,0)),(-1, 1)).asnumpy()
    label = reshape(label[1:].transpose((1,2,0)),(-1, 1)).asnumpy()
    predict = reshape(predict[1:].transpose((1,2,0)),(-1, 1)).asnumpy()

    average_precision = dict()
    # for i in range(n_classes):
    #     precision[i], recall[i], _ = precision_recall_curve(label[:, i], predict[:, i])
        # average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
    precision, recall, _ = precision_recall_curve(label, predict)
    return precision, recall

def get_Precision(label, predict):
    # label & predict:[3,512,512]
    C, H, W = label.shape
    reshape = ops.Reshape()

    label = reshape(label.transpose((1,2,0)),(H * W, C))
    predict = reshape(predict.transpose((1,2,0)),(H * W, C))
    metric = nn.Precision('multilabel')
    metric.clear()
    metric.update(label, predict)
    p = metric.eval(average=True)
    # N,C 中 N 是样本数， C 是类别数
    return p

def get_Recall(label, predict):
    # label & predict:[3,512,512]
    C, H, W = label.shape
    reshape = ops.Reshape()
    label = reshape(label.transpose((1,2,0)),(H * W, C))
    # print(label.shape)  #(262144,3)
    predict = reshape(predict.transpose((1,2,0)),(H * W, C))
    metric = nn.Recall('multilabel')
    metric.clear()
    metric.update(label, predict)
    recall = metric.eval(average=True)
    # N,C 中 N 是样本数， C 是类别数
    return recall

def test(network,ckpt_path):
    # network = TransUNet(3,3)
    valid_dataset_generator = DatasetGenerator(
        cfg.DATA_DIR,cfg.VALID_IMG,cfg.VALID_MASK)
    valid_dataset = ds.GeneratorDataset(valid_dataset_generator,
                                        ["image", "label"],
                                        shuffle=False)
    valid_dataset = valid_dataset.batch(1, num_parallel_workers=1)

    param_dict = load_checkpoint(ckpt_path)# 加载权重文件
    load_param_into_net(network, param_dict)# 权重传入网络

    # 开始测试
    print("============== Starting Testing ============")
    miou_total, dice_total = 0, 0
    for n, (image, label) in enumerate(tqdm(valid_dataset)):# valid dataset
        if n > 100:
            break
        predict = network(image.astype(np.float32))#get output [1,3,512,512]
        squeeze = ops.Squeeze(0)
        predict = squeeze(predict)  #[3,512,512]
        mask_path = squeeze(label) #[3,512,512]

        iou = get_iou(mask_path,predict)
        miou_total += iou
        dice = get_dice(mask_path,predict)
        dice_total += dice
    print(f'miou value is {miou_total / (n+1) }\n average dice value is {dice_total / (n+1)}')

def test_single(network,ckpt_path):
    # network = TransUNet(3,3)
    valid_dataset_generator = DatasetGenerator(
        cfg.DATA_DIR,cfg.VALID_IMG,cfg.VALID_MASK)
    valid_dataset = ds.GeneratorDataset(valid_dataset_generator,
                                        ["image", "label"],
                                        shuffle=False)
    valid_dataset = valid_dataset.batch(1, num_parallel_workers=1)

    param_dict = load_checkpoint(ckpt_path)# 加载权重文件
    load_param_into_net(network, param_dict)# 权重传入网络

    # 开始测试
    print("============== Starting Testing ============")
    # logging.info('============== Starting Testing ============')
    dice = 0
    iou = 0
    liver_iou = 0
    lesion_iou = 0
    liver_dice = 0
    lesion_dice = 0
    lesion_count = 0

    for n, (image, label) in tqdm(enumerate(tqdm(valid_dataset))):# valid dataset
        if n > 100:
            break
        predict = network(image.astype(np.float32))#get output [1,3,512,512]
        squeeze = ops.Squeeze(0)
        predict = squeeze(predict)  #[3,512,512]
        mask = squeeze(label) #[3,512,512]

        b_iou = get_iou(mask, predict)
        b_dice = get_dice(mask, predict)
        print(f'batch iou {b_iou}, batch dice {b_dice}')
        # logging.info(f'batch iou {b_iou}, batch dice {b_dice}')
        iou += b_iou
        dice += b_dice

        b_iou = get_iou(mask[1], predict[1])
        b_dice = get_dice(mask[1], predict[1])
        print(f'LIVER: batch iou {b_iou}, batch dice {b_dice}')
        # logging.info(f'LIVER: batch iou {b_iou}, batch dice {b_dice}')
        liver_iou += b_iou
        liver_dice += b_dice

        b_iou = get_iou(mask[2], predict[2])
        b_dice = get_dice(mask[2], predict[2])
        print(f'LESION: batch iou {b_iou}, batch dice {b_dice}')
        # logging.info(f'LESION: batch iou {b_iou}, batch dice {b_dice}')
        lesion_iou += b_iou
        lesion_dice += b_dice

        if b_iou > 0:
            lesion_count += 1
    
    # logging.info('============end single evaluate====================')
    
    print(f'TOTAL:        avg iou {iou / (n+1)}, avg dice {dice / (n+1)}')
    # logging.info(f'        avg iou {iou / (n+1)}, avg dice {dice / (n+1)}')
    print(f'LIVER:  avg iou {liver_iou / (n+1)}, avg dice {liver_dice / (n+1)}')
    # logging.info(f'LIVER:  avg iou {liver_iou / (n+1)}, avg dice {liver_dice / (n+1)}')
    print(f'LESION: avg iou {lesion_iou / lesion_count}, avg dice {lesion_dice / lesion_count}')
    # logging.info(f'LESION: avg iou {lesion_iou / lesion_count}, avg dice {lesion_dice / lesion_count}')
    # logging.info(f'total amount pictures: {n+1}, total lesion pictures:{lesion_count}')

def logger_config(log_path,logging_name):
    '''
    配置log
    :param log_path: 输出log路径
    :param logging_name: 记录中name，可随意
    :return:
    '''
    '''
    logger是日志对象，handler是流处理器，console是控制台输出（没有console也可以，将不会在控制台输出，会在日志文件中输出）
    '''
    # 获取logger对象,取名
    logger = logging.getLogger(logging_name)
    # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
    logger.setLevel(level=logging.DEBUG)
    # 获取文件日志句柄并设置日志级别，第二层过滤
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    # 生成并设置文件日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # 为logger对象添加句柄
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

def test_presion_recall(network,ckpt_path):
    valid_dataset_generator = DatasetGenerator(
        cfg.DATA_DIR,cfg.VALID_IMG,cfg.VALID_MASK)
    valid_dataset = ds.GeneratorDataset(valid_dataset_generator,
                                        ["image", "label"],
                                        shuffle=False)
    valid_dataset = valid_dataset.batch(1, num_parallel_workers=1)

    param_dict = load_checkpoint(ckpt_path)# 加载权重文件
    load_param_into_net(network, param_dict)# 权重传入网络

    # 开始测试
    print("============== Starting Testing ============")
    p , r = [], []
    for n, (image, label) in tqdm(enumerate(tqdm(valid_dataset))):# valid dataset
        if n > 100:
            break
        predict = network(image.astype(np.float32))#get output [1,3,512,512]
        squeeze = ops.Squeeze(0)
        predict = squeeze(predict)  #[3,512,512]
        label = squeeze(label) #[3,512,512]

        predict[predict > 0.5] = 1
        predict[predict <= 0.5] = 0
        # recall = get_Recall(label, predict)
        # precision = get_Precision(label, predict)
        
        precision, recall = get_pr(label, predict)
        print(f'recall {recall} precision {precision}')
        p.append(precision[1])
        r.append(recall[1])
    print(f'total img valid is {n}')
    print(f'aver precision is {np.mean(p)},aver recall is {np.mean(r)}')

if __name__ == "__main__":
    from src.attnunet import AttU_Net
    ckpt_path = '/public/users/WUT/wut.gaoyl03/zengxiang/beta4.0/output_ckpts/best_newtransunet_d1_hybridloss.ckpt'
    # network = UNet(3,3)
    # network = AttU_Net(3,3)
    network = TransUNet(3,3)
    model = ckpt_path.split('/')[-1].split('.')[0]
    # test_single(network,ckpt_path)
    test_presion_recall(network,ckpt_path)

    # logger = logger_config(log_path='./metircs_logs/' + str(model) + '_ep10.log', logging_name='')
    
    # x = Tensor(np.random.random([3,512,512]))
    # x[x > 0.5] = 1
    # x[x <= 0.5] = 0
    # y = Tensor(np.ones([3,512,512]))
    # r = get_Recall(x,y)
    # r = get_Precision(x, y)
    # u,c = np.unique(x.asnumpy(),return_counts=True)
    # print(u,c)
    # u1,c1 = np.unique(r,return_counts=True)
    # print(r)
    # print(f'====={u1},{c1}')


