import mindspore as msp
import mindspore.nn as nn
from mindspore import Tensor,ops

class DoubleConv(nn.Cell):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()

        self.doubel_conv = nn.SequentialCell(
            nn.Conv2d(in_ch, out_ch, 3, has_bias=True),
            nn.BatchNorm2d(out_ch), 
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, has_bias=True),
            nn.BatchNorm2d(out_ch), 
            nn.ReLU())

    def construct(self, x):
        x = self.doubel_conv(x)
        return x


class up_conv(nn.Cell):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.SequentialCell(
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,pad_mode='pad', padding=1,has_bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU()
        )
        self.upsample = nn.ResizeBilinear()

    def construct(self,x):
        x = self.upsample(x,scale_factor=2)
        x = self.up(x)
        return x

class Attention_block(nn.Cell):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.SequentialCell(
            nn.Conv2d(F_g, F_int, kernel_size=1,has_bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.SequentialCell(
            nn.Conv2d(F_l, F_int, kernel_size=1, has_bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.SequentialCell(
            nn.Conv2d(F_int, 1, kernel_size=1, has_bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def construct(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return x * psi


class AttU_Net(nn.Cell):
    def __init__(self, img_ch=3, output_ch=3):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = DoubleConv(img_ch, 64)
        self.Conv2 = DoubleConv(64, 128)
        self.Conv3 = DoubleConv(128, 256)
        self.Conv4 = DoubleConv(256, 512)
        self.Conv5 = DoubleConv(512, 1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = DoubleConv(1024, 512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = DoubleConv(512,256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = DoubleConv(256, 128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = DoubleConv(128, 64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, has_bias=True)
        self.sigmoid = nn.Sigmoid()
        self.cat = ops.Concat(axis=1)

    def construct(self, x):
        # 3 ,512 ,512
        # encoding path
        x1 = self.Conv1(x)#64 512 512

        x2 = self.Maxpool(x1)# 64 256 256
        x2 = self.Conv2(x2) #128 256 256

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)#256 128 128

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)#512 64 64

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)#1024 32 32

        # decoding + concat path
        d5 = self.Up5(x5)   # 512 64 64
        x4 = self.Att5(g=d5, x=x4)
        d5 = self.cat((x4, d5))
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = self.cat((x3, d4))
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = self.cat((x2, d3))
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = self.cat((x1, d2))
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.sigmoid(d1)

        return d1


# 测试函数
if __name__ == '__main__':

    import numpy as np
    img_size = 512
    x = msp.Tensor(np.ones([4,3,img_size,img_size]), msp.float32)
    model = AttU_Net(3,3)
    output = model(x)   # output.shape (B,3,512,512)
    print(output)
    print(output.shape)