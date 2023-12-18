from mindspore import nn
from mindspore.ops import constexpr
import math
import numpy as np
import mindspore.numpy as numpy
import mindspore.common.dtype as mstype
from mindspore import nn
import mindspore as msp
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
import mindspore.ops as ops

@constexpr
def compute_kernel_size(inp_shape, output_size):
    kernel_width, kernel_height = inp_shape[2], inp_shape[3]
    if isinstance(output_size, int):
        kernel_width = math.ceil(kernel_width / output_size) 
        kernel_height = math.ceil(kernel_height / output_size)
    elif isinstance(output_size, list) or isinstance(output_size, tuple):
        kernel_width = math.ceil(kernel_width / output_size[0]) 
        kernel_height = math.ceil(kernel_height / output_size[1])
    return (kernel_width, kernel_height)


class double_conv(nn.Cell):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.double_conv = nn.SequentialCell(
            nn.Conv2d(in_ch, out_ch, 3),
            nn.BatchNorm2d(out_ch), 
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3),
            nn.BatchNorm2d(out_ch), 
            nn.ReLU())

    def construct(self, x):
        x = self.double_conv(x)
        return x

class AdaptiveMaxPool2d(nn.Cell):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
    
    def construct(self, x):
        inp_shape = x.shape
        kernel_size = compute_kernel_size(inp_shape, self.output_size)
        return ops.MaxPool(kernel_size, kernel_size)(x)


class UNet_3Plus_DeepSup(nn.Cell):

    def __init__(self, 
                in_channels, 
                n_classes):
        super(UNet_3Plus_DeepSup, self).__init__()
        self.in_channels = in_channels # 输入通道
        # self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]

        ## -------------Encoder--------------
        self.conv1 = double_conv(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = double_conv(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = double_conv(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = double_conv(filters[2], filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # [N,512,32,32]->[N,1024,32,32]
        self.conv5 = double_conv(filters[3], filters[4])

        # # 加入一个Transformer Encoder
        # # 层数
        # layers = []
        # dpr = [x.item() for x in np.linspace(0, drop_path_rate, depth)]
        # for i in range(depth):
        #     layers.append(TransEncoder(dim=embedding_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #                         dropout=dropout, attn_dropout=attn_dropout, drop_connect=dpr[i]))
        # self.blocks = nn.SequentialCell(layers)
        

        self.sigmoid = nn.Sigmoid()

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks #64* 5=320

        '''stage 4d'''
        # h1:64,512,512 -> 64,64,64 -> 64,64,64
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, pad_mode='pad',padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU()

        # h2:128,256,256 -> 128,64,64 ->64,64,64 same padding
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, pad_mode='pad',padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU()

        # h3:256,128,128 ->256,64,64 ->64,64,64
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, pad_mode='pad',padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU()

        # h4: 512,64,64 ->64,64,64
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, pad_mode='pad',padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU()

        # hd5:1024,32,32 -> 1024,64,64 ->64,64,64
        self.hd5_UT_hd4 = nn.ResizeBilinear()  
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, pad_mode='pad',padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU()

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, pad_mode='pad',padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU()

        '''stage 3d'''
        # h1:64,512,512 -> 64,128,128 ->64,128,128
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, pad_mode='pad',padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU()

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, pad_mode='pad',padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU()

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, pad_mode='pad',padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU()

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.ResizeBilinear()  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, pad_mode='pad',padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU()

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.ResizeBilinear()  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, pad_mode='pad',padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU()

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, pad_mode='pad',padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU()

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, pad_mode='pad',padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU()

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, pad_mode='pad',padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU()

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.ResizeBilinear()  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, pad_mode='pad',padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU()

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.ResizeBilinear()  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, pad_mode='pad',padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU()

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.ResizeBilinear()  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, pad_mode='pad',padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU()

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, pad_mode='pad',padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU()

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, pad_mode='pad',padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU()

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.ResizeBilinear()  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, pad_mode='pad',padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU()

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.ResizeBilinear()  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, pad_mode='pad',padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU()

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.ResizeBilinear()  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, pad_mode='pad',padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU()

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.ResizeBilinear()  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, pad_mode='pad',padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU()

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, pad_mode='pad',padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU()

        # -------------Bilinear Upsampling--------------
        self.upscore5 = nn.ResizeBilinear()
        self.upscore4 = nn.ResizeBilinear()
        self.upscore3 = nn.ResizeBilinear()
        self.upscore2 = nn.ResizeBilinear()

        # DeepSup
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, pad_mode='pad',padding=1)
        self.outconv2 = nn.Conv2d(self.UpChannels, n_classes, 3, pad_mode='pad',padding=1)
        self.outconv3 = nn.Conv2d(self.UpChannels, n_classes, 3, pad_mode='pad',padding=1)
        self.outconv4 = nn.Conv2d(self.UpChannels, n_classes, 3, pad_mode='pad',padding=1)
        self.outconv5 = nn.Conv2d(filters[4], n_classes, 3, pad_mode='pad',padding=1)

    

    # def dotProduct(self,seg,cls):
    #     B, N, H, W = seg.shape
    #     seg = self.reshape(seg,(B, N, H * W))
        
    #     # ijk, ij -> ijk
    #     seg = self.transpose(seg, (2,0,1))         
    #     tmp = self.multiply(seg, cls)
    #     final = self.transpose(tmp, (1,2,0))
    #     # final = torch.einsum("ijk,ij->ijk", [seg, cls])  #11111
    #     final = self.reshape(final,(B, N, H, W))
    #     return final


    def construct(self, inputs):
        # input: 3,512,512
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->64,512,512

        h2 = self.maxpool1(h1)  #64,256,256
        h2 = self.conv2(h2)     # h2->128,256,256

        h3 = self.maxpool2(h2)  #128,128,128
        h3 = self.conv3(h3)     # h3->256,128,128

        h4 = self.maxpool3(h3)  #256,64,64
        h4 = self.conv4(h4)  # h4->512,64,64

        h5 = self.maxpool4(h4)  #512,32,32
        hd5 = self.conv5(h5)  # h5->1024,32,32

        # # -------------Classification-------------
        # cls_branch = self.cls(hd5)[:,:,0,0]  #(1, 1024, 32,32)--> (B,N,1,1)->(B,N)
        # # shape (4,2) (B,) Tensor
        # cls_branch_max = cls_branch.argmax(axis=1)  # N diem max value
        # # [0 0 0 0] shape(4,) 
        # cls_branch_max = cls_branch_max[:, None] # (B)->(B,1)

        # --------------TransEncoder-------------
        # hd5 = self.blocks(hd5) + hd5 # (1, 1024, 32,32)


        ## -------------Decoder-------------
        ## h1:64,512,512 -> 64,64,64 -> 64,64,64
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        ## h2:128,256,256 -> 128,64,64 ->64,64,64 same padding
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        ## h3:256,128,128 ->256,64,64 ->64,64,64
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        # # h4: 512,64,64 ->64,64,64
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        # hd5:1024,32,32 -> 1024,64,64 ->64,64,64
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5,scale_factor=2))))
        # 64*5=320,64,64 ->320,64,64
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            numpy.concatenate((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)))) # hd4->40*40*UpChannels

        # h1:64,512,512 -> 64,128,128 ->64,128,128
        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        # h2:128,256,256 ->64,128,128 ->64,128,128
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        # h3:256,128,128 ->64,128,128
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        # h4:320,64,64 -> 320,128,128 ->  64,128,128
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4,scale_factor=2))))
        # 1024,32,32 ->1024,128,128 ->64,128,128
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5,scale_factor=4))))
        # 64*5=320,128,128 ->320,128,128
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            numpy.concatenate((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3,scale_factor=2))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4,scale_factor=4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5,scale_factor=8))))
        # 64*5=320,256,256->320,256,256
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            numpy.concatenate((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2,scale_factor=2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3,scale_factor=4))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4,scale_factor=8))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5,scale_factor=16))))
        # 64*5=320,512,512->320,512,512
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            numpy.concatenate((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # hd1->320*320*UpChannels

        ## deep supervision 侧输出
        d5 = self.outconv5(hd5)#1024,32,32->2,32,32
        d5 = self.upscore5(d5,scale_factor=16) #2,32,32->2,512,512

        d4 = self.outconv4(hd4) #320,64,64 ->2,64,64
        d4 = self.upscore4(d4,scale_factor=8) # 2,512,512

        d3 = self.outconv3(hd3) #320,128,128 ->2,128,128
        d3 = self.upscore3(d3,scale_factor=4) # 2,512,512

        d2 = self.outconv2(hd2) #320,236,256 ->2,256,256
        d2 = self.upscore2(d2,scale_factor=2) 

        d1 = self.outconv1(hd1) # 320,512,512->2,512,512

        # d1 = self.dotProduct(d1, cls_branch_max)
        # d2 = self.dotProduct(d2, cls_branch_max)
        # d3 = self.dotProduct(d3, cls_branch_max)
        # d4 = self.dotProduct(d4, cls_branch_max)
        # d5 = self.dotProduct(d5, cls_branch_max)

        # return self.sigmoid(d1)
        return self.sigmoid(d1), self.sigmoid(d2), self.sigmoid(d3), self.sigmoid(d4), self.sigmoid(d5)



'''
    UNet 3+ with deep supervision and class-guided module
'''
class UNet_3Plus_DeepSup_CGM(nn.Cell):

    def __init__(self, 
                in_channels, 
                n_classes):
        super(UNet_3Plus_DeepSup_CGM, self).__init__()
        self.in_channels = in_channels # 输入通道
        # self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]

        ## -------------Encoder--------------
        self.conv1 = double_conv(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = double_conv(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = double_conv(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = double_conv(filters[2], filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # [N,512,32,32]->[N,1024,32,32]
        self.conv5 = double_conv(filters[3], filters[4])

        # # 加入一个Transformer Encoder
        # # 层数
        # layers = []
        # dpr = [x.item() for x in np.linspace(0, drop_path_rate, depth)]
        # for i in range(depth):
        #     layers.append(TransEncoder(dim=embedding_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #                         dropout=dropout, attn_dropout=attn_dropout, drop_connect=dpr[i]))
        # self.blocks = nn.SequentialCell(layers)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks #64* 5=320

        '''stage 4d'''
        # h1:64,512,512 -> 64,64,64 -> 64,64,64
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, pad_mode='pad',padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU()

        # h2:128,256,256 -> 128,64,64 ->64,64,64 same padding
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, pad_mode='pad',padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU()

        # h3:256,128,128 ->256,64,64 ->64,64,64
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, pad_mode='pad',padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU()

        # h4: 512,64,64 ->64,64,64
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, pad_mode='pad',padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU()

        # hd5:1024,32,32 -> 1024,64,64 ->64,64,64
        self.hd5_UT_hd4 = nn.ResizeBilinear()  
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, pad_mode='pad',padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU()

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, pad_mode='pad',padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU()

        '''stage 3d'''
        # h1:64,512,512 -> 64,128,128 ->64,128,128
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, pad_mode='pad',padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU()

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, pad_mode='pad',padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU()

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, pad_mode='pad',padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU()

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.ResizeBilinear()  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, pad_mode='pad',padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU()

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.ResizeBilinear()  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, pad_mode='pad',padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU()

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, pad_mode='pad',padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU()

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, pad_mode='pad',padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU()

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, pad_mode='pad',padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU()

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.ResizeBilinear()  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, pad_mode='pad',padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU()

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.ResizeBilinear()  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, pad_mode='pad',padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU()

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.ResizeBilinear()  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, pad_mode='pad',padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU()

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, pad_mode='pad',padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU()

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, pad_mode='pad',padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU()

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.ResizeBilinear()  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, pad_mode='pad',padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU()

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.ResizeBilinear()  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, pad_mode='pad',padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU()

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.ResizeBilinear()  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, pad_mode='pad',padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU()

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.ResizeBilinear()  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, pad_mode='pad',padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU()

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, pad_mode='pad',padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU()

        # -------------Bilinear Upsampling--------------
        self.upscore6 = nn.ResizeBilinear()###uesless
        self.upscore5 = nn.ResizeBilinear()
        self.upscore4 = nn.ResizeBilinear()
        self.upscore3 = nn.ResizeBilinear()
        self.upscore2 = nn.ResizeBilinear()

        # DeepSup
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, pad_mode='pad',padding=1)
        self.outconv2 = nn.Conv2d(self.UpChannels, n_classes, 3, pad_mode='pad',padding=1)
        self.outconv3 = nn.Conv2d(self.UpChannels, n_classes, 3, pad_mode='pad',padding=1)
        self.outconv4 = nn.Conv2d(self.UpChannels, n_classes, 3, pad_mode='pad',padding=1)
        self.outconv5 = nn.Conv2d(filters[4], n_classes, 3, pad_mode='pad',padding=1)

        self.cls = nn.SequentialCell(
                    nn.Dropout(keep_prob=0.5),
                    nn.Conv2d(filters[4], 2, 1),
                    AdaptiveMaxPool2d(1),
                    nn.Sigmoid())
        self.transpose = ops.Transpose()
        self.multiply = ops.Mul()
        self.reshape = P.Reshape()
        self.sigmoid = nn.Sigmoid()

        # # initialise weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init_weights(m, init_type='kaiming')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         init_weights(m, init_type='kaiming')

    def dotProduct(self,seg,cls):
        B, N, H, W = seg.shape
        seg = self.reshape(seg,(B, N, H * W))
        
        # ijk, ij -> ijk
        seg = self.transpose(seg, (2,0,1))         
        tmp = self.multiply(seg, cls)
        final = self.transpose(tmp, (1,2,0))
        # final = torch.einsum("ijk,ij->ijk", [seg, cls])  #11111
        final = self.reshape(final,(B, N, H, W))
        return final


    def construct(self, inputs):
        # input: 3,512,512
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->64,512,512

        h2 = self.maxpool1(h1)  #64,256,256
        h2 = self.conv2(h2)     # h2->128,256,256

        h3 = self.maxpool2(h2)  #128,128,128
        h3 = self.conv3(h3)     # h3->256,128,128

        h4 = self.maxpool3(h3)  #256,64,64
        h4 = self.conv4(h4)  # h4->512,64,64

        h5 = self.maxpool4(h4)  #512,32,32
        hd5 = self.conv5(h5)  # h5->1024,32,32

        # -------------Classification-------------
        cls_branch = self.cls(hd5)[:,:,0,0]  #(1, 1024, 32,32)--> (B,N,1,1)->(B,N)
        # shape (4,2) (B,) Tensor
        cls_branch_max = cls_branch.argmax(axis=1)  # N diem max value
        # [0 0 0 0] shape(4,) 
        cls_branch_max = cls_branch_max[:, None] # (B)->(B,1)

        # --------------TransEncoder-------------
        # hd5 = self.blocks(hd5) + hd5 # (1, 1024, 32,32)


        ## -------------Decoder-------------
        ## h1:64,512,512 -> 64,64,64 -> 64,64,64
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        ## h2:128,256,256 -> 128,64,64 ->64,64,64 same padding
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        ## h3:256,128,128 ->256,64,64 ->64,64,64
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        # # h4: 512,64,64 ->64,64,64
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        # hd5:1024,32,32 -> 1024,64,64 ->64,64,64
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5,scale_factor=2))))
        # 64*5=320,64,64 ->320,64,64
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            numpy.concatenate((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)))) # hd4->40*40*UpChannels

        # h1:64,512,512 -> 64,128,128 ->64,128,128
        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        # h2:128,256,256 ->64,128,128 ->64,128,128
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        # h3:256,128,128 ->64,128,128
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        # h4:320,64,64 -> 320,128,128 ->  64,128,128
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4,scale_factor=2))))
        # 1024,32,32 ->1024,128,128 ->64,128,128
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5,scale_factor=4))))
        # 64*5=320,128,128 ->320,128,128
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            numpy.concatenate((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3,scale_factor=2))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4,scale_factor=4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5,scale_factor=8))))
        # 64*5=320,256,256->320,256,256
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            numpy.concatenate((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2,scale_factor=2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3,scale_factor=4))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4,scale_factor=8))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5,scale_factor=16))))
        # 64*5=320,512,512->320,512,512
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            numpy.concatenate((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # hd1->320*320*UpChannels

        ## deep supervision 侧输出
        d5 = self.outconv5(hd5)#1024,32,32->2,32,32
        d5 = self.upscore5(d5,scale_factor=16) #2,32,32->2,512,512

        d4 = self.outconv4(hd4) #320,64,64 ->2,64,64
        d4 = self.upscore4(d4,scale_factor=8) # 2,512,512

        d3 = self.outconv3(hd3) #320,128,128 ->2,128,128
        d3 = self.upscore3(d3,scale_factor=4) # 2,512,512

        d2 = self.outconv2(hd2) #320,236,256 ->2,256,256
        d2 = self.upscore2(d2,scale_factor=2) 

        d1 = self.outconv1(hd1) # 320,512,512->2,512,512

        d1 = self.dotProduct(d1, cls_branch_max)
        d2 = self.dotProduct(d2, cls_branch_max)
        d3 = self.dotProduct(d3, cls_branch_max)
        d4 = self.dotProduct(d4, cls_branch_max)
        d5 = self.dotProduct(d5, cls_branch_max)

        # return self.sigmoid(d1)
        return self.sigmoid(d1), self.sigmoid(d2), self.sigmoid(d3), self.sigmoid(d4), self.sigmoid(d5)


# 测试函数
if __name__ == '__main__':

    import mindspore as msp
    img_size = 512
    x = msp.Tensor(np.ones([4,3,img_size,img_size]), msp.float32)
    # y = Tensor(np.ones([4,3]), msp.float32)
    model = UNet_3Plus_DeepSup(3,3)
    # dot = model.dotProduct(x,y)
    # print(dot)
    # print(dot.shape)
    output = model(x)   # output.shape (B,3,512,512)
    # print(output)
    for i in range(5):
        print(f'{i} output shape is:{output[i].shape} \n')
    # print(type(output))