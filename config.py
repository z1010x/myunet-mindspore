# step1：定义数据集DataLoader
# step2：定义忘了UNet
# step3：定义损失函数
# step4：定义优化器
# step5：定义训练流程

from easydict import EasyDict as edict
__C = edict()
# Consumers can get config by: import config as cfg
cfg = __C

# For dataset dir
__C.DATA_DIR = '/public/users/WUT/wut.gaoyl03/zengxiang/beta5.0/LITS_IMAGE'  #划分好的数据集
__C.TRAIN_IMG = 'train/ct'              #训练图片集
__C.TRAIN_MASK = 'train/seg'              #训练标签集
__C.VALID_IMG = 'val/ct'                #验证图片集
__C.VALID_MASK = 'val/seg'                #验证标签集

# For training
__C.BATCH_SIZE = 8 #batch_size???
__C.EPOCHS = 20   #epoch
__C.SAVE_CHECKPOINT_STEPS = 1000 #1000个step存一个模型文件
__C.KEEP_CHECKPOINT_MAX = 20    #最多保存的文件数目
__C.OUTPUT_DIR = './output_train/unet_416_lr'#输出训练好的模型
__C.CKPT_PRE_TRAINED = './output_train'#预训练模型，紧接着之前的进行训练
__C.LOSS_SCALE = 1024.0 #loss规模管理
__C.EVAL_PER_EPOCH = 1  #每个epoch测试一次
# __C.LR_TYPE = 'poly'#学习率的变化方式，可以设置为'poly'、'cos'、'exp'

# For lr
__C.LR_TYPE = 'exp'#学习率的变化方式，可以设置为'poly'、'cos'、'exp'
__C.BASE_LR = 1e-5 #学习率的初始值
__C.LR_DECAY_STEP = 10000 #学习率的变化的步数
__C.LR_DECAY_RATE = 0.1   #学习率的变化速率
# 保持平稳lr 再改 实验所得

