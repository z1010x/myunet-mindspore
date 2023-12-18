from mindspore import nn
import mindspore.ops as ops
from mindspore.common.initializer import initializer,HeUniform
import mindspore as msp

class DoubleConv(nn.Cell):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()

        self.DoubleConv = nn.SequentialCell(
            # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, 
            # padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            nn.Conv2d(in_ch, out_ch, 3, has_bias=True, weight_init=HeUniform()),
            # torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, 
            # affine=True, track_running_stats=True)
            nn.BatchNorm2d(out_ch), 
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, has_bias=True, weight_init=HeUniform()),
            nn.BatchNorm2d(out_ch), 
            nn.ReLU())

    def construct(self, x):
        x = self.DoubleConv(x)
        return x


class UNET(nn.Cell):    # 输入为B,3，256，256
    # TEST:input = Tensor(np.ones([1,3,256,256]),mindspore.float32)
    def __init__(self, in_ch, out_ch):
        super(UNET, self).__init__()
        # Encoder
        # [N,3,256,256]->[N,64,256,256]
        self.DoubleConv1 = DoubleConv(in_ch, 32)
        # [N,64,256,256]->[N,64,128,128]

        # torch.nn.MaxPool2d(kernel_size=2, stride=None,
        # padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [N,64,128,128]->[N,128,128,128]
        self.DoubleConv2 = DoubleConv(32, 64)
        # [N,128,128,128]->[N,128,64,64]
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [N,128,64,64]->[N,256,64,64]
        self.DoubleConv3 = DoubleConv(64, 128)
        # [N,256,64,64]->[N,256,32,32]
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [N,256,32,32]->[N,512,32,32]
        self.DoubleConv4 = DoubleConv(128, 256)
        # [N,512,32,32]->[N,512,16,16]
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [N,512,16,16]->[N,1024,16,16]
        self.DoubleConv5 = DoubleConv(256, 512)

        # Decoder
        # [N,1024,16,16]->[N,1024,32,32]
        # torch: self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        # torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
        # stride=1, padding=0, output_padding=0, groups=1, bias=True, 
        # dilation=1, padding_mode='zeros')
        self.upsample1 = nn.Conv2dTranspose(512, 256, 2, stride=2, has_bias=True, weight_init=HeUniform())
        # [N,1024+512,32,32]->[N,512,32,32]
        self.DoubleConv6 = DoubleConv(512, 256)
        # [N,512,32,32]->[N,512,64,64]
   
        self.upsample2 = nn.Conv2dTranspose(256, 128, 2, stride=2, has_bias=True, weight_init=HeUniform())
        # [N,512+256,64,64]->[N,256,64,64]
        self.DoubleConv7 = DoubleConv(256, 128)
        # [N,256,64,64]->[N,256,128,128]
        self.upsample3 = nn.Conv2dTranspose(128, 64, 2, stride=2, has_bias=True, weight_init=HeUniform())
        # [N,256+128,128,128]->[N,128,128,128]
        self.DoubleConv8 = DoubleConv(128, 64)
        # [N,128,128,128]->[N,128,256,256]
        self.upsample4 = nn.Conv2dTranspose(64, 32, 2, stride=2, has_bias=True, weight_init=HeUniform())
        # [N,128+64,256,256]->[N,64,256,256]
        self.DoubleConv9 = DoubleConv(64, 32)
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, 
        # padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.final = nn.Conv2d(32, out_ch, 1, has_bias=True, weight_init=HeUniform())
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)
        self.cat = ops.Concat(axis=1)
       

    def construct(self, x):
        feature1 = self.DoubleConv1(x)
        tmp = self.maxpool1(feature1)
        feature2 = self.DoubleConv2(tmp)
        tmp = self.maxpool2(feature2)
        feature3 = self.DoubleConv3(tmp)
        tmp = self.maxpool3(feature3)
        feature4 = self.DoubleConv4(tmp)
        tmp = self.maxpool4(feature4)
        feature5 = self.DoubleConv5(tmp)

        up_feature1 = self.upsample1(feature5)
        merge6 = self.cat([feature4, up_feature1])
        c6 = self.DoubleConv6(merge6)
        up_feature2 = self.upsample2(c6)
        merge7 = self.cat([feature3, up_feature2])
        c7 = self.DoubleConv7(merge7)
        up_feature3 = self.upsample3(c7)
        merge8 = self.cat([feature2, up_feature3])
        c8 = self.DoubleConv8(merge8)
        up_feature4 = self.upsample4(c8)
        merge9 = self.cat([feature1, up_feature4])
        c9 = self.DoubleConv9(merge9)
        output = self.final(c9)
        output = self.sigmoid(output)
        # output = self.softmax(output)

        return output
        # output.shape(1,2,512,512)

# 测试函数
if __name__ == '__main__':

    import numpy as np
    img_size = 512
    x = msp.Tensor(np.ones([4,3,img_size,img_size]), msp.float32)
    model = UNET(3,3)
    output = model(x)   # output.shape (B,3,512,512)
    print(output)
    print(output.shape)