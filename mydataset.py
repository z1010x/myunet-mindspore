import os
import numpy as np
import cv2
import mindspore as ms
from mindspore import ops, nn
from mindspore import dtype as mstype
import mindspore.dataset as ds
import mindspore.common.initializer as init
from pre_process import data_augment
from config import cfg

# 自定义数据集
class DatasetGenerator:
    def __init__(self, root_dir, img_dir, label_dir):
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_path = os.listdir(os.path.join(self.root_dir, self.img_dir))
        self.label_path = os.listdir(os.path.join(self.root_dir, self.label_dir))

    def __getitem__(self, index):
        # 读取img
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.root_dir, self.img_dir, img_name)
        img = cv2.imread(img_item_path, cv2.IMREAD_COLOR)

        # 读取label
        label_name = self.label_path[index]
        label_item_path = os.path.join(self.root_dir, self.label_dir,label_name)
        label = cv2.imread(label_item_path, cv2.IMREAD_GRAYSCALE)

        # # 进行数据增强
        img, label = data_augment(img, label)
        img = np.transpose(img, (2, 0, 1))  # img.shape (3,512,512)
   
        label = label // 86 #得到标签类
        label = label.astype(np.int)
        # label 0 background 1 liver 2 lesion
        label = (np.arange(3) == label[..., None]).astype(int)# 512,512,3
        label = np.transpose(label,(2,0,1)) # (3,512,512)
        
        img = (img / 255.).astype(np.float32)
        label = (label / 1.).astype(np.float32)

        return img, label   

    def __len__(self):
        return len(self.img_path)

