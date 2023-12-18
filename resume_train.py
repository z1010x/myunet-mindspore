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
from mindspore.train.callback import LossMonitor, TimeMonitor, ModelCheckpoint, CheckpointConfig
from mindspore import context
import loss as myloss
from src.transunet import TransUNet
from src.unet import UNet
from src.transunetpp import TransUNet_CGM
from src.unet_in_unet import Unet_in_Unet
from src.transunet_3plus import TransUNet_3Plus_DeepSup_CGM
from src.unet_3plus import UNet_3Plus_DeepSup_CGM, UNet_3Plus_DeepSup
# import learning_rates
from config import cfg
from mindspore import load_checkpoint, load_param_into_net, ops
import learning_rates


def train(ckpt_path):
    # 生成数据集
    train_dataset_generator = DatasetGenerator(
        cfg.DATA_DIR, cfg.TRAIN_IMG,cfg.TRAIN_MASK)
    # 构建数据集
    train_dataset = ds.GeneratorDataset(train_dataset_generator,
                                        ["image", "label"],
                                        shuffle=True)
    # 验证集
    valid_dataset_generator = DatasetGenerator(
        cfg.DATA_DIR,cfg.VALID_IMG,cfg.VALID_MASK)
    valid_dataset = ds.GeneratorDataset(valid_dataset_generator,
                                        ["image", "label"],
                                        shuffle=True)

    # 打包Batch_size=2
    train_dataset = train_dataset.batch(cfg.BATCH_SIZE, num_parallel_workers=1)
    # valid_dataset = valid_dataset.batch(cfg.BATCH_SIZE, num_parallel_workers=1)

    
    loss = myloss.MyBCELoss()
    # loss = nn.BCELoss(reduction='mean')

    # train_net = Unet_in_Unet(3,3)
    # train_net = UNet_3Plus_DeepSup(3,3)
    train_net = TransUNet(3,3)
   

    # ckpt_path = '/home/zhulifu/zxf/all_411/output_train/unet/ckpt_unet-1_7665.ckpt'
    param_dict = load_checkpoint(ckpt_path)# 加载权重文件
    load_param_into_net(train_net, param_dict)# 权重传入网络
    # print(f'net type {type(train_net)}')

    optimizer = nn.Adam(train_net.trainable_params(), learning_rate=1e-7, weight_decay=1e-8)

    # # load pretrained model
    # if cfg.CKPT_PRE_TRAINED:
    #     param_dict = load_checkpoint(cfg.CKPT_PRE_TRAINED)
    #     load_param_into_net(train_net, param_dict)

    # optimizer
    iters_per_epoch = train_dataset.get_dataset_size()
    
    # 总共训练步长
    total_train_steps = iters_per_epoch * cfg.EPOCHS
 
    print("dataset length is:", iters_per_epoch)
    print('total_train_steps',total_train_steps)

   
    config_ckpt = CheckpointConfig(
        save_checkpoint_steps=iters_per_epoch,
        keep_checkpoint_max=50)

    # 修改配置信息文件名
    # prefix表示生成CheckPoint文件的前缀名；directory：表示存放模型的目录
    ckpoint_cb = ModelCheckpoint(prefix='ckpt_transunet',
                            directory='./output_train/transunet_lr_dynamic',
                            config=config_ckpt)
    # callback for saving ckpts
    time_cb = TimeMonitor(data_size=iters_per_epoch)
    loss_cb = LossMonitor()

    cbs = [time_cb, loss_cb, ckpoint_cb]

     # 定义model 
    # amp_level = "O0" if context.get_context("device_target") == "GPU" else "O3"
    model = Model(train_net,
                  optimizer=optimizer,
                  loss_fn=loss)

    # 使用model.train接口进行训练
    print("============== Starting Training ==============")
    print("Model:",type(train_net))
    print('loss type is ', type(loss))
    print("Device:",context.get_context("device_target"))
    
    print("dataset length is:", iters_per_epoch)
    print('total_train_steps',total_train_steps)

    print("Config",cfg)
    try:
        model.train(cfg.EPOCHS, train_dataset, callbacks=cbs, dataset_sink_mode=False)  
    except KeyboardInterrupt:
        print('interrupted')
    
    print("============== End Training ==============")


if __name__ == "__main__":    
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    # ckpt 20ep lr:1e-4 bs4 nn.bceloss
    # train('/public/users/WUT/wut.gaoyl03/zengxiang/beta4.0/output_train/transunet_161/ckpt_transunet-20_3833.ckpt')
    # ckpt 10ep lr:1e-5 bs1 nn.bceloss
    train('/public/users/WUT/wut.gaoyl03/zengxiang/beta4.0/output_train/transunet_161/ckpt_transunet-20_3833.ckpt')