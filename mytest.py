from mindspore import load_checkpoint, load_param_into_net
import mindspore as msp
from src.transunet import TransUNet
from src.unet import UNet
import numpy as np
import mindspore.numpy as numpy
import os
import cv2
import mindspore.nn as nn
from mindspore import ops,Tensor
from config import cfg
# from tqdm import tqdm
from mydataset import DatasetGenerator
import mindspore.dataset as ds
import mindspore.ops as ops
from mindspore import context


## 定义验证函数
class dice_coeff(nn.Metric):
    def __init__(self):#初始化
        super(dice_coeff, self).__init__()
        self.clear()

    # 值重置为0
    def clear(self):
        self._dice_coeff_sum = 0
        self._samples_num = 0

    # 更新dice值
    def update(self, *inputs):#*input represent (y_pred, y)
        if len(inputs) != 2:
            raise ValueError('Mean dice coeffcient need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))
        # 转换为ndarray
        # y_pred = self._convert_data(inputs[0])

        # y_pred = inputs[0]  # predict (b,3,512,512)
        # # print(f'y pred shape {y_pred.shape}')#（1，3，512，512）
        # # y = self._convert_data(inputs[1])
        # y = inputs[1]   # label (b,3,512,512)
        # # print(f'y shape {y.shape}')#（1，3，512，512）
        # # 样本数 B
        # self._samples_num += y.shape[0] #   +4
        # # 预测转置 ==》B,H,W,C
        # # (1,512，512,3)
        # y_pred = y_pred.transpose(0, 2, 3, 1)
        # # y_pred = y_pred[0]
        # y = y.transpose(0, 2, 3, 1)
        # y = y[0]
        # 得到每个batchsize的第一个
        # 不改变shape大小 数值进行归一化输出（0，1）
        # sigmoid=ops.Sigmoid()
        # y_pred = sigmoid(y_pred)
        ###################################3
        '''
        网络中加入softmax操作
        '''

        # y *= 122    # 进行可视化展现需要
        #data-->ndarry
        # y = self._convert_data(y)
        # y_pred = self._convert_data(y_pred)
        # for index in range(y_pred.shape[0]):
        #     for channel in range(4):
        #         cv2.imwrite(f'./pictures/ypred_{index}_c_{channel}.png', y_pred[index,:,:, channel] * 255)
        #         if 1 in np.unique(y[index, :, :, channel]) and channel != 0:
        #             print(f'./pictures/ylabel_{index}_c_{channel}.png')
        #         cv2.imwrite(f'./pictures/ylabel_{index}_c_{channel}.png', y[index, :, :, channel] * 255)
        # 交集 、并集
        single_dice_coeff = 0
        y_pred = inputs[0].asnumpy()
        y = inputs[1].asnumpy()
        for index in range(y_pred.shape[0]):
            inter = np.dot(y_pred[index].flatten(), y[index].flatten())
            union = np.dot(y_pred[index].flatten(), y_pred[index].flatten()) + np.dot(y[index].flatten(), y[index].flatten())
            # single_dice_coeff 
            single_dice_coeff += 2 * float(inter) / float(union + 1e-6)
        self._samples_num += y.shape[0]
        self._dice_coeff_sum += single_dice_coeff
    
        return single_dice_coeff / y.shape[0]
        # 一批的dice值
        

    # 验证函数
    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self._dice_coeff_sum / float(self._samples_num)


def test(network,pic_dir, ckpt_path):
    #定义网络
    # network = UNet(3,3)
    
    valid_dataset_generator = DatasetGenerator(
        cfg.DATA_DIR,cfg.VALID_IMG,cfg.VALID_MASK)
    valid_dataset = ds.GeneratorDataset(valid_dataset_generator,
                                        ["image", "label"],
                                        shuffle=False)
    valid_dataset = valid_dataset.batch(1, num_parallel_workers=1)

    # ckpt_path = cfg.OUTPUT_DIR
    # # 得到文件名
    # ckpt_list = os.listdir(ckpt_path)
    # # 结果以字典形式保存
    # result = dict()
    # for ckpt in tqdm(ckpt_list):
    #     cnt = 0
    #     fwiou = 0.
    #     # 加载具体的某一个ckpt文件
    #     param_dict = load_checkpoint(os.path.join(ckpt_path, ckpt))
    #     # # 将参数加载到网络中
    #     sigmoid=ops.Sigmoid()
    #     load_param_into_net(network, param_dict)

    #     # 定义损失函数 验证集 和model
    #     criterion = WeightedBCELoss(w0=1.39, w1=1.69)

    #     # loss scale manager
    #     manager_loss_scale = FixedLossScaleManager(cfg.LOSS_SCALE,drop_overflow_update=False)
    #     # model = Model(net, loss_fn=criterion,loss_scale_manager=loss_scale_manager, metrics={"dice_coeff": dice_coeff()})

    #     # 定义model
    #     model = Model(network,
    #                 eval_network=network,
    #                 # loss_fn=criterion,
    #                 # loss_scale_manager=manager_loss_scale,
    #                 metrics={"dice_coeff": dice_coeff()}
    #                 # metrics={"accuracy"}
    #                 )
    # 单个ckpt
  
    param_dict = load_checkpoint(ckpt_path)# 加载权重文件
    load_param_into_net(network, param_dict)# 权重传入网络


    # model.eval开始验证
    print("============== Starting Evaluating ============")
    # pic_dir = './pictures_unet_ep1'
    if not os.path.exists(pic_dir):
        os.mkdir(pic_dir)
    # dice_score = model.eval(valid_dataset, dataset_sink_mode=False)
    dice = 0
    for n, (image, label) in enumerate(valid_dataset):# from 0
        if n > 16:   # bs = 1 valid dataset
            break
        # print(index,image,label)
        # 通过Softmax函数就可以将多分类的输出值转换为范围在[0, 1]和为1的概率分布。
        # sigmoid=ops.Sigmoid()
        # relu = nn.ReLU()
        # softmax = nn.Softmax()
        out = network(image.astype(np.float32))
        # out = softmax(out)# 得到输出out 归一化到（0-1）
        # out = out * 255
        # print(out[0])
        # print(label[0])    #94,3,512,512
        dc = dice_coeff()   #进入验证函数
        single_dice_coeff = dc.update(out, label)# 计算返回单个dice值
        # print(f'index = {n} ,single_dice_coeff = {single_dice_coeff}')

        # op = ops.Concat()
        # label = label.transpose(0, 2, 3, 1)#4,3,512,512
        # for index in range(out.shape[0]):# index = 0-3

        cv2.imwrite(f'{pic_dir}/{n}_image.png', (image[0,:,:,:].transpose(1,2,0) * 255).asnumpy())
        cv2.imwrite(f'{pic_dir}/{n}_label.png', (label[0,:,:,:].transpose(1,2,0) * 255).asnumpy())
        cv2.imwrite(f'{pic_dir}/{n}_pred.png', (out[0,:,:,:].transpose(1,2,0) * 255).asnumpy())
        
    #     dice += dc.eval()

    # print(f'total dice:{dice / (n) }')
    # predict pictures:   pictures_unet_ep1     total dice:0.9296235828716105 




def draw_dataset(pic_dir,size=4):
    import numpy as np
    import matplotlib.pyplot as plt
    # pic_dir = './pictures_unet'
    num = len(os.listdir(pic_dir)) // 3
    # print(f'num is {num}')
    file_list = [(f'{pic_dir}/{index}_image.png', f'{pic_dir}/{index}_label.png', f'{pic_dir}/{index}_pred.png') for index in range(num)]
    # print(f'file-list length:{len(file_list)}')
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
    plt.savefig('./results/' + pic_dir.split('/')[-1] + '.png')


if __name__ =="__main__":

    ## 可视化展现
    # context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    # network = TransUNet(3,3)
    network = UNet(3,3)
    ckpt_path = '/public/users/WUT/wut.gaoyl03/zengxiang/beta4.0/output_train/unet_lr-5/ckpt_unet-17_1917.ckpt'
    pic_dir_suf = ckpt_path.split('/')[-1].split('.')[0]
    pic_dir =  './pictures_' + pic_dir_suf
    test(network,pic_dir,ckpt_path)
    draw_dataset(pic_dir =pic_dir)
        
    