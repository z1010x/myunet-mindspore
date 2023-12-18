import numpy as np
import mindspore.numpy as numpy
import mindspore.common.dtype as mstype
from mindspore import nn
import mindspore as msp
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
import mindspore.ops as ops

# unet-双卷积
class double_conv(nn.Cell):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.double_conv = nn.SequentialCell(
            nn.Conv2d(in_ch, out_ch, 3, has_bias=True),
            nn.BatchNorm2d(out_ch), 
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, has_bias=True),
            nn.BatchNorm2d(out_ch), 
            nn.ReLU())

    def construct(self, x):
        x = self.double_conv(x)
        return x


class TransUNet(nn.Cell):    # 输入为B,3，256，256
    # TEST:input = Tensor(np.ones([1,3,256,256]),mindspore.float32)
    def __init__(self,
            in_ch,
            out_ch,
            img_size=512,
            patch_size=16,
            in_channels=3,
            embedding_dim=32 * 32,# 111
            num_heads=8,
            mlp_ratio=4,
            qkv_bias=False,
            num_class=1000,
            stride=4,
            dropout=0.5,
            attn_dropout=0.5,
            drop_path_rate=0.5,
            depth=1):
        super(TransUNet, self).__init__()
        # Encoder
        # [N,3,256,256]->[N,32,256,256]
        self.double_conv0 = double_conv(in_ch, 32)
        # [n,32,256,256]->[n,64,256,256]
        self.double_conv1 = double_conv(32, 64)
        # [N,64,256,256]->[N,64,128,128]
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [N,64,128,128]->[N,128,128,128]
        self.double_conv2 = double_conv(64, 128)
        # [N,128,128,128]->[N,128,64,64]
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [N,128,64,64]->[N,256,64,64]
        self.double_conv3 = double_conv(128, 256)
        # [N,256,64,64]->[N,256,32,32]
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [N,256,32,32]->[N,512,32,32]
        self.double_conv4 = double_conv(256, 512)
        # [N,512,32,32]->[N,512,16,16]
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [N,512,16,16]->[N,1024,16,16]
        self.double_conv5 = double_conv(512, 1024)

        # 加入一个Transformer Encoder
        layers = []
        dpr = [x.item() for x in np.linspace(0, drop_path_rate, depth)]
        for i in range(depth):
            layers.append(TransEncoder(dim=embedding_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                dropout=dropout, attn_dropout=attn_dropout, drop_connect=dpr[i]))
        self.blocks = nn.SequentialCell(layers)
        
        # Decoder
        # [N,1024,16,16]->[N,1024,32,32]
        self.upsample1 = nn.Conv2dTranspose(1024, 512, 2, stride=2, has_bias=True)
        # [N,1024+512,32,32]->[N,512,32,32]
        self.double_conv6 = double_conv(1024, 512)
        # [N,512,32,32]->[N,512,64,64]
        self.upsample2 = nn.Conv2dTranspose(512, 256, 2, stride=2, has_bias=True)
        # [N,512+256,64,64]->[N,256,64,64]
        self.double_conv7 = double_conv(512, 256)
        # [N,256,64,64]->[N,256,128,128]
        self.upsample3 = nn.Conv2dTranspose(256, 128, 2, stride=2, has_bias=True)
        # [N,256+128,128,128]->[N,128,128,128]
        self.double_conv8 = double_conv(256, 128)
        # [N,128,128,128]->[N,128,256,256]
        self.upsample4 = nn.Conv2dTranspose(128, 64, 2, stride=2, has_bias=True)
        # [N,128+64,256,256]->[N,64,256,256]
        self.double_conv9 = double_conv(128, 64)
        self.double_conv10 = double_conv(64,32)
        self.final = nn.Conv2d(32, out_ch, 1,has_bias=True)
        self.softmax = nn.Softmax(1)
        self.sigmoid = nn.Sigmoid()
        self.cat = ops.Concat(axis=1)
        # 3类输出[background liver lesion ]

    def construct(self, x):
        x = self.double_conv0(x)
        feature1 = self.double_conv1(x)
        tmp = self.maxpool1(feature1)
        feature2 = self.double_conv2(tmp)
        tmp = self.maxpool2(feature2)
        feature3 = self.double_conv3(tmp)
        tmp = self.maxpool3(feature3)
        feature4 = self.double_conv4(tmp)
        tmp = self.maxpool4(feature4)
        feature5 = self.double_conv5(tmp)

        feature5 = self.blocks(feature5) + feature5 # (1, 1024, 32,32)
        
        up_feature1 = self.upsample1(feature5)
        merge6 = self.cat([feature4, up_feature1])
        c6 = self.double_conv6(merge6)
        up_feature2 = self.upsample2(c6)
        merge7 = self.cat([feature3, up_feature2])
        c7 = self.double_conv7(merge7)
        up_feature3 = self.upsample3(c7)
        merge8 = self.cat([feature2, up_feature3])
        c8 = self.double_conv8(merge8)
        up_feature4 = self.upsample4(c8)
        merge9 = self.cat([feature1, up_feature4])
        c9 = self.double_conv9(merge9)

        c10 = self.double_conv10(c9)#b,32,512,512
        output = self.final(c10)
        output = self.sigmoid(output)

        return output
        # # output.shape(1,3,256,256)

class MLP(nn.Cell):
    """MLP"""

    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.dropout = nn.Dropout(1. - dropout)
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.act = nn.GELU()

    def construct(self, x):
        """MLP"""
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Cell):
    """Multi-head Attention"""

    def __init__(self, dim, hidden_dim=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        hidden_dim = hidden_dim or dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Dense(dim, hidden_dim * 3, has_bias=qkv_bias)
        self.softmax = nn.Softmax(axis=-1)
        self.batmatmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.attn_drop = nn.Dropout(1. - attn_drop)
        self.batmatmul = P.BatchMatMul()
        self.proj = nn.Dense(hidden_dim, dim)
        self.proj_drop = nn.Dropout(1. - proj_drop)

        self.transpose = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, x):
        """Multi-head Attention"""
        B, N, _ = x.shape
        qkv = self.transpose(self.reshape(self.qkv(x), (B, N, 3, self.num_heads, self.head_dim)), (2, 0, 3, 1, 4))
        # qkv = self.transpose(self.reshape(self.qkv(x), (B, N, 3, 8, 32)), (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = self.softmax(self.batmatmul_trans_b(q, k) * self.scale)
        attn = self.attn_drop(attn)
        x = self.reshape(self.transpose(self.batmatmul(attn, v), (0, 2, 1, 3)), (B, N, -1))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DropConnect(nn.Cell):
    """drop connect implementation"""

    def __init__(self, drop_connect_rate=0., seed=0):
        super(DropConnect, self).__init__()
        self.keep_prob = 1 - drop_connect_rate
        seed = min(seed, 0) # always be 0
        self.rand = P.UniformReal(seed=seed) # seed must be 0, if set to other value, it's not rand for multiple call
        self.shape = P.Shape()
        self.floor = P.Floor()

    def construct(self, x):
        """drop connect implementation"""
        if self.training:
            x_shape = self.shape(x) # B N C
            random_tensor = self.rand((x_shape[0], 1, 1))
            random_tensor = random_tensor + self.keep_prob
            random_tensor = self.floor(random_tensor)
            x = x / self.keep_prob
            x = x * random_tensor
        return x


class TransEncoder(nn.Cell):
    """Transfomer Encoder Block"""
    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias=False, dropout=0., attn_dropout=0., drop_connect=0.):
        super(TransEncoder, self).__init__()
        # transformer
        self.norm1 = nn.LayerNorm([dim])
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_dropout,
                              proj_drop=dropout)
        self.drop_connect = DropConnect(drop_connect)
        self.norm2 = nn.LayerNorm([dim])
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=dropout)
        # aug path
        self.augs_attn = nn.Dense(dim, dim, has_bias=True)
        self.augs = nn.Dense(dim, dim, has_bias=True)
        self.reshape = P.Reshape()
        self.tile = P.Tile()
        self.concat = P.Concat(axis=1)

    def construct(self, x):
        """augvit Block"""
        B,C,H,W = x.shape
        x = self.reshape(x, (B, 1024, -1))

        x_norm1 = self.norm1(x)
        x = x + self.drop_connect(self.attn(x_norm1)) + self.augs_attn(x_norm1)
        x_norm2 = self.norm2(x)
        x = x + self.drop_connect(self.mlp(x_norm2)) + self.augs(x_norm2)
        x = self.reshape(x, (B, 1024, 32, 32))# 111
        return x


class PatchEmbed(nn.Cell):
    """Image to Patch Embedding"""

    def __init__(self, img_size, patch_size=16, in_channels=3, embedding_dim=768):
        super(PatchEmbed, self).__init__()
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size, has_bias=True)
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()

    def construct(self, x):
        """Image to Patch Embedding"""
        x = self.proj(x) # B, 768, 14, 14
        B, C, H, W = x.shape
        x = self.reshape(x, (B, C, H * W))
        x = self.transpose(x, (0, 2, 1)) # B, N, C
        return x



# 测试函数
if __name__ == '__main__':
    import cv2
    import os
    from mindspore import context
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU", save_graphs=False)
    img_path = '/public/users/WUT/wut.gaoyl03/zengxiang/LITS_IMAGE/ct/volume-0_slice_46.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = np.transpose(img, (2, 0, 1)) # 3,512,512
    img = (img / 255.).astype(np.float32)
    # img_size = 512
    # x = Tensor(np.random.random([4,3,img_size,img_size]), msp.float32)
    x = Tensor(img)
    x = x.expand_dims(axis=0)
    print(x[0][0])
    u, c = np.unique(img[0], return_counts=True)
    # print(u,c)
    print('==============================')
    model = TransUNet(3,3)
    output = model(x)   # output.shape (B,3,512,512)
    print(output[0][0])
    print(output.shape)
    y = output.asnumpy()
    u1, c1 = np.unique(y[0], return_counts=True)
    print(u1, c1)

