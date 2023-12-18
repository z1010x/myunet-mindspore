## TransUNet模型开发

#### 介绍

​         TranUNet 将具有全局自我注意力机制的 Transformer 技术与基础 U-Net 相结合成TranUNet 的网络架构。TransUNet 从序列到序列的预测角度建立了自我注意力机制，加入到 U-Net 网络的深层次特征中，网络经过对 TransBlock 编码之后的自注意特征进行上采样操作，并与编码器上同一分辨率的高语义特征进行连接融合操作。TransUNet 不仅保留了基础 U-Net 的优秀定位能力，更凸显了 Transformer 自适应定位目标区域的优势，使得模型更好的关注所感兴趣的肝肿瘤区域。TransUNet 在基础 U-Net 的结构上进行如下改进，更好的利用了注意力机制。

<h4>模型架构</h4>

![image-20220718214026847](C:\Users\28237\AppData\Roaming\Typora\typora-user-images\image-20220718214026847.png)


#### 安装教程

- CUDA11.3

- Install MindSpore：MindSpore  1.7.0

- Install PyTorch：Pytorch 1.11.0 

- ```
  pip install -r requirements.txt
  ```

<h4>代码目录结构</h4>

```
MyUNet
    │  config.py
    │  learning_rates.py
    │  loss.py
    │  metrics.py
    │  mydataset.py
    │  mytest.py
    │  plt_msp.py
    │  predict.py
    │  pre_process.py
    │  README.en.md
    │  README.md
    │  resume_train.py
    │  train.py
    │  util.py
    │
    ├─.gitee
    │      ISSUE_TEMPLATE.zh-CN.md
    │      PULL_REQUEST_TEMPLATE.zh-CN.md
    │
    ├─log
    ├─output_train
    └─src
            attnunet.py
            new_unet.py
            resunet.py
            transunet.py
            transunet3plus.py
            transunet_3plus.py
            unet.py
            unet_3plus.py
```

#### 训练

```
python train.py
```

<h4>结果</h4>

![image-20220718215605122](C:\Users\28237\AppData\Roaming\Typora\typora-user-images\image-20220718215605122.png)

![image-20220718215627682](C:\Users\28237\AppData\Roaming\Typora\typora-user-images\image-20220718215627682.png)