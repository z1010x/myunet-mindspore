import os
from cv2 import imwrite
import mindspore
import mindspore.dataset as ds
from mindspore import Tensor
import numpy as np
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset.transforms.py_transforms import Compose
from mydataset import DatasetGenerator
from mindspore import nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.callback import TimeMonitor, ModelCheckpoint, CheckpointConfig
from mindspore import context
import loss as myloss
from src.transunet import TransUNet
from src.unet import UNet
from src.new_unet import UNET
from config import cfg
from mindspore import load_checkpoint, load_param_into_net, ops
import learning_rates
import logging
from src.transunet_3plus import TransUNet_3Plus_DeepSup
from src.unet_3plus import UNet_3Plus_DeepSup_CGM, UNet_3Plus_DeepSup
from src.attnunet import AttU_Net
from src.resunet import ResUnet

from mindvision.engine.callback import ValAccMonitor,LossMonitor


def train(train_net,BATCH_SIZE,EPOCHS,BASE_LR):
    # 生成数据集
    DATA_DIR = '/public/users/WUT/wut.gaoyl03/zengxiang/beta5.0/LITS_IMAGE'  #划分好的数据集
    TRAIN_IMG = 'train/ct'              #训练图片集
    TRAIN_MASK = 'train/seg'              #训练标签集
    VALID_IMG = 'val/ct'                #验证图片集
    VALID_MASK = 'val/seg'  

    train_dataset_generator = DatasetGenerator(
        DATA_DIR, TRAIN_IMG,TRAIN_MASK)
    # 构建数据集
    train_dataset = ds.GeneratorDataset(train_dataset_generator,
                                        ["image", "label"],
                                        shuffle=True)
    # 验证集
    valid_dataset_generator = DatasetGenerator(
        DATA_DIR,VALID_IMG,VALID_MASK)
    valid_dataset = ds.GeneratorDataset(valid_dataset_generator,
                                        ["image", "label"],
                                        shuffle=True)

    # 打包Batch_size=4
    train_dataset = train_dataset.batch(BATCH_SIZE, num_parallel_workers=1)
    valid_dataset = valid_dataset.batch(1, num_parallel_workers=1)

    ## msp-gpu==>1.6.1
    # loss = nn.BCELoss(reduction='mean')
    loss = myloss.MyHybridLoss()
    
    ## if deep supervision
    # loss = myloss.MyBCELoss()
    # optimizer
    iters_per_epoch = train_dataset.get_dataset_size()
    
    # 总共训练步长
    total_train_steps = iters_per_epoch * EPOCHS
    
    print("dataset length is:", iters_per_epoch)
    print('total_train_steps',total_train_steps)


    # lr_iter = nn.NaturalExpDecayLR(BASE_LR, 0.9, iters_per_epoch, True)

    optimizer = nn.Adam(train_net.trainable_params(),BASE_LR)
    # optimizer = nn.SGD(train_net.trainable_params(),BASE_LR)

    # # load pretrained model
    # if cfg.CKPT_PRE_TRAINED:
    #     param_dict = load_checkpoint(cfg.CKPT_PRE_TRAINED)
    #     load_param_into_net(train_net, param_dict)

    # # 保存模型
    # config_ckpt = CheckpointConfig(
    #     save_checkpoint_steps=iters_per_epoch,
    #     keep_checkpoint_max=50)
    # # 修改配置信息文件名
    # # prefix表示生成CheckPoint文件的前缀名；directory：表示存放模型的目录
    # ckpoint_cb = ModelCheckpoint(prefix='ckpt_'+ ckpt_predix,
    #                         directory='./output_train/' + output_dir,
    #                         config=config_ckpt)

    amp_level = "O0" if context.get_context("device_target") == "GPU" else "O3"
    model = Model(train_net,
                  optimizer=optimizer,
                  loss_fn=loss,
                  amp_level=amp_level,
                  metrics={"Accuracy": nn.Dice(smooth=1e-5)})
    # callback for saving ckpts
    time_cb = TimeMonitor(data_size=iters_per_epoch)
    loss_cb = LossMonitor(lr_init=BASE_LR)
    val_cb = ValAccMonitor(model, valid_dataset, num_epochs=EPOCHS, 
                    ckpt_directory='./output_ckpts', best_ckpt_name='best_transunet.ckpt')

    cbs = [time_cb, loss_cb, val_cb]
    
    # model.train(cfg.EPOCHS, train_dataset, callbacks=cbs, dataset_sink_mode=False)
    # 使用model.train接口进行训练
    print("============== Starting Training ==============")
    print("Model:",type(train_net))
    print('loss type is ', type(loss))
    print('optimizer is: ',type(optimizer))
    # print('lr type is :',type(lr_iter))
    print("Device:",context.get_context("device_target"))
    print(f"BATCH_SIZE={BATCH_SIZE},EPOCHS={EPOCHS},BASE_LR={BASE_LR}")

    model.train(EPOCHS, train_dataset, callbacks=cbs, dataset_sink_mode=False)
    print("============== End Training ==============")
    print("dataset length is:", iters_per_epoch)
    print('total_train_steps',total_train_steps)


if __name__ == "__main__":    
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    # train_net = AttU_Net(3,3)
    # train_net = ResUnet(3)
    # train_net = UNET(3,3)
    train_net = TransUNet(3,3)
    # train_net = UNet_3Plus_DeepSup(3,3)
    ckpt_predix = 'transunet'
    output_dir = 'transunet'
       
    ## 开始训练，参数在main里面给
    train(train_net,BATCH_SIZE=4,EPOCHS=12,BASE_LR = 1e-5)