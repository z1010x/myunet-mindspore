from genericpath import exists
from mindspore import load_checkpoint, load_param_into_net
import mindspore as msp
from src.transunet import TransUNet
from src.unet import UNet
from src.new_unet import UNET
from src.attnunet import AttU_Net

import numpy as np
import mindspore.numpy as numpy
import os
import cv2
import mindspore.nn as nn
from mindspore import ops,Tensor
from config import cfg
from tqdm import tqdm
from mydataset import DatasetGenerator
import mindspore.dataset as ds
import mindspore.ops as ops
from mindspore import context


def test(pic_dir, ckpt_path):
    #定义网络
    # network = UNet(3,3)
    network = TransUNet(3,3)
    # network = UNET(3,3)
    # network = AttU_Net(3,3)
    valid_dataset_generator = DatasetGenerator(
        cfg.DATA_DIR,cfg.VALID_IMG,cfg.VALID_MASK)
    valid_dataset = ds.GeneratorDataset(valid_dataset_generator,
                                        ["image", "label"],
                                        shuffle=False)
    valid_dataset = valid_dataset.batch(1, num_parallel_workers=1)
  
    param_dict = load_checkpoint(ckpt_path)# 加载权重文件
    load_param_into_net(network, param_dict)# 权重传入网络


    # model.eval开始验证
    print("============== Starting Evaluating ============")
    # pic_dir = './pictures_unet_ep1'
    if not os.path.exists(pic_dir):
        os.mkdir(pic_dir)
    for n, (image, label) in tqdm(enumerate(valid_dataset)):# from 0
        if n > 16:   # bs = 1 valid dataset
            break
        out = network(image.astype(np.float32))
        # cv2.imwrite(f'{pic_dir}/{n}_image.png', (image[0,:,:,:].transpose(1,2,0) * 255).asnumpy())
        # cv2.imwrite(f'{pic_dir}/{n}_label.png', (label[0,:,:,:].transpose(1,2,0) * 255).asnumpy())
        # cv2.imwrite(f'{pic_dir}/{n}_pred.png', (out[0,:,:,:].transpose(1,2,0) * 255).asnumpy())

        cv2.imwrite(f'{pic_dir}/{n}_image.png', (image[0,:,:,:].transpose(1,2,0) * 255).asnumpy())
        label = label[0].transpose(1,2,0)   # label h w c
        cv2.imwrite(f'{pic_dir}/{n}_label.png', ((0 * label[:,:,0]+1 * label[:,:,1]+2* label[:,:,2] )* 122).asnumpy())
        # (0 * label[0]+1 * label[1]+2* label[2] )* 122
        out = out[0].transpose(1,2,0)  
        cv2.imwrite(f'{pic_dir}/{n}_pred.png', ((0 * out[:,:,0]+1 * out[:,:,1]+4* out[:,:,2] )* 122).asnumpy())
        




def draw_dataset(pic_dir,size=4):
    import numpy as np
    import matplotlib.pyplot as plt
    # pic_dir = './pictures_unet'
    num = len(os.listdir(pic_dir)) // 3
    print(f'num is {num-1}')
    file_list = [(f'{pic_dir}/{index}_image.png', f'{pic_dir}/{index}_label.png', f'{pic_dir}/{index}_pred.png') for index in range(num)]
    print(f'file-list length:{len(file_list)-1}')
    fig = plt.figure(figsize=(15,8))#整体大小
    rows = size
    cols = size
    # img = np.zeros((rows*28, cols*28))
    # for i in range(rows):
    #     for j in range(cols):
    #         img[i*28:(i+1)*28, j*28:(j+1)*28] = ds[i][j].numpy()
    # plt.imsave('img.png', img, cmap='Greys')
    # plt.imshow(img, cmap='Greys')

    
    axes=[]
    # fig = plt.figure()
    # fig = plt.figure(figsize=(12, 12))
    
    # for a in range(rows*cols):
    # cv2.imread()读取彩色图像后得到的格式是BGR格式，像素值范围在0~255之间,通道格式为HWC
    for i in range(rows):
        for j in range(cols):
            # b = np.random.randint(7, size=(height,width))
            # b = ds[i][j].numpy()
            x, l, p = file_list[size * i + j]
            x = cv2.imread(x)
            l = cv2.imread(l)
            p = cv2.imread(p)
            # 蓝底
            # x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            # l = cv2.cvtColor(l, cv2.COLOR_BGR2RGB)
            # p = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)

            hline = np.zeros((512,1,3), dtype=np.int64)
            b = np.concatenate((x, l, hline, p), axis=1)
            axes.append( fig.add_subplot(rows, cols, cols*i+j+1) )
            subplot_title=('image                  label                  predict')
            axes[-1].set_title(subplot_title)
            plt.axis('off')
            plt.imshow(b, cmap="Greys")
    
    fig.tight_layout()
    # ep = pic_dir.split('/')[-1].split('-')[-1].split('_')[0]
    filename = './results/' + pic_dir.split('/')[-1] + '.png'
    plt.savefig(filename)
    print('predict results are saved in ' + filename)


if __name__ =="__main__":

    ## 可视化展现
    # context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    ckpt_path = '/public/users/WUT/wut.gaoyl03/zengxiang/beta4.0/output_ckpts/best_newtransunet_d1_hybridloss.ckpt'
    pic_dir_suf = ckpt_path.split('/')[-1].split('.')[0]
    # model_name = ckpt_path.split('/')[-2]
    pic_dir = os.path.join('./output_predict', pic_dir_suf)

    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)

    test(pic_dir,ckpt_path)
    draw_dataset(pic_dir =pic_dir)
        
    