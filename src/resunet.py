import mindspore as msp
import mindspore.nn as nn
from mindspore import Tensor,ops

class ResidualConv(nn.Cell):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()
        self.conv_block = nn.SequentialCell(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, pad_mode='pad', padding=padding),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, pad_mode='pad', padding=1),
        )
        self.conv_skip = nn.SequentialCell(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, pad_mode='pad', padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def construct(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Cell):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()
        self.upsample = nn.Conv2dTranspose(input_dim, output_dim, kernel_size=kernel, stride=stride)

    def construct(self, x):
        return self.upsample(x)



class ResUnet(nn.Cell):
    def __init__(self, channel, filters=[64, 128, 256, 512]):
        super(ResUnet, self).__init__()

        self.input_layer = nn.SequentialCell(
            nn.Conv2d(channel, filters[0], kernel_size=3, pad_mode='pad', padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, pad_mode='pad', padding=1),
        )
        self.input_skip = nn.SequentialCell(
            nn.Conv2d(channel, filters[0], kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU()
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.SequentialCell(
            nn.Conv2d(filters[0], 3, 1, 1),
            nn.Sigmoid(),
        )
        self.cat = ops.Concat(axis=1)

    def construct(self, x):
        # Encode x: 3512 512
        x1 = self.input_layer(x) + self.input_skip(x)   #64 512 512 + 64 512 512
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)   # 4 256 128 128
        # Bridge
        x4 = self.bridge(x3)   
        # Decode
        x4 = self.upsample_1(x4)
        x5 = self.cat([x4, x3])

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = self.cat([x6, x2])

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = self.cat([x8, x1])

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return output

# 测试函数
if __name__ == '__main__':

    import numpy as np
    img_size = 512
    x = msp.Tensor(np.ones([4,3,img_size,img_size]), msp.float32)
    model = ResUnet(3)
    output = model(x)   # output.shape (B,3,512,512)
    # print(output)
    print(output.shape)