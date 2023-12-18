from mindspore import nn
import mindspore
import mindspore.numpy as np
import mindspore.ops as ops
from mindspore import Tensor

import mindspore
import mindspore.nn as nn
import mindspore.common.dtype as mstype
import mindspore.ops.operations as F
from mindspore.ops import functional as F2
from mindspore.nn.cell import Cell


class MyBCELoss(nn.Cell):

    def __init__(self, weight=None, reduction='none'):
        """Initialize BCELoss."""
        super(MyBCELoss, self).__init__()
        self.bceloss = nn.BCELoss()
        

    def construct(self, logits, labels):
        loss = 0
        num = len(logits)
        for output in logits:
            loss += self.bceloss(output,labels)
        loss = loss / num

        return loss



class MyLoss(Cell):
    """
    Base class for other losses.
    """
    def __init__(self, reduction='mean'):
        super(MyLoss, self).__init__()
        if reduction is None:
            reduction = 'none'

        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"reduction method for {reduction.lower()} is not supported")

        self.average = True
        self.reduce = True
        if reduction == 'sum':
            self.average = False
        if reduction == 'none':
            self.reduce = False

        self.reduce_mean = F.ReduceMean()
        self.reduce_sum = F.ReduceSum()
        self.mul = F.Mul()
        self.cast = F.Cast()

    def get_axis(self, x):
        shape = F2.shape(x)
        length = F2.tuple_len(shape)
        perm = F2.make_range(0, length)
        return perm

    def get_loss(self, x, weights=1.0):
        """
        Computes the weighted loss
        Args:
            weights: Optional `Tensor` whose rank is either 0, or the same rank as inputs, and must be broadcastable to
                inputs (i.e., all dimensions must be either `1`, or the same as the corresponding inputs dimension).
        """
        input_dtype = x.dtype
        x = self.cast(x, mstype.float32)
        weights = self.cast(weights, mstype.float32)
        x = self.mul(weights, x)
        if self.reduce and self.average:
            x = self.reduce_mean(x, self.get_axis(x))
        if self.reduce and not self.average:
            x = self.reduce_sum(x, self.get_axis(x))
        x = self.cast(x, input_dtype)
        return x

    def construct(self, base, target):
        raise NotImplementedError


from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.primitive import constexpr
from mindspore.nn.layer.activation import get_activation
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel


class MyLogSoftmax(nn.Cell):
    def __init__(self, axis=-1):
        """Initialize LogSoftmax."""
        super(MyLogSoftmax, self).__init__()
        self.softmax = P.Softmax(axis)
        
    def construct(self, x):
        return ops.log(self.softmax(x))

class MyFocalLoss(nn.LossBase):

    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        """Initialize MyFocalLoss."""
        super(MyFocalLoss, self).__init__(reduction=reduction)

        self.gamma = validator.check_value_type("gamma", gamma, [float])
        if weight is not None and not isinstance(weight, Tensor):
            raise TypeError(f"For '{self.cls_name}', the type of 'weight' should be a Tensor, "
                            f"but got {type(weight).__name__}.")
        if isinstance(weight, Tensor) and weight.ndim != 1:
            raise ValueError(f"For '{self.cls_name}', the dimension of 'weight' should be 1, but got {weight.ndim}.")
        self.weight = weight
        self.expand_dims = P.ExpandDims()
        self.gather_d = P.GatherD()
        self.squeeze = P.Squeeze(axis=1)
        self.tile = P.Tile()
        self.cast = P.Cast()
        self.dtype = P.DType()
        self.logsoftmax = MyLogSoftmax(1)

    def construct(self, logits, labels):
        # _check_is_tensor('logits', logits, self.cls_name)
        # _check_is_tensor('labels', labels, self.cls_name)
        labelss = labels
        # _check_ndim(logits.ndim, labelss.ndim)
        # _check_channel_and_shape(logits.shape[1], labelss.shape[1])
        # _check_input_dtype(self.dtype(labelss), self.cls_name)

        if logits.ndim > 2:
            logits = logits.view(logits.shape[0], logits.shape[1], -1)
            labelss = labelss.view(labelss.shape[0], labelss.shape[1], -1)
        else:
            logits = self.expand_dims(logits, 2)
            labelss = self.expand_dims(labelss, 2)

        log_probability = self.logsoftmax(logits)

        if labels.shape[1] == 1:
            log_probability = self.gather_d(log_probability, 1, self.cast(labelss, mindspore.int32))
            log_probability = self.squeeze(log_probability)

        probability = F.exp(log_probability)

        if self.weight is not None:
            convert_weight = self.weight[None, :, None]
            convert_weight = self.tile(convert_weight, (labelss.shape[0], 1, labelss.shape[2]))
            if labels.shape[1] == 1:
                convert_weight = self.gather_d(convert_weight, 1, self.cast(labelss, mindspore.int32))
                convert_weight = self.squeeze(convert_weight)
            log_probability = log_probability * convert_weight

        weight = F.pows(-1 * probability + 1.0, self.gamma)
        if labels.shape[1] == 1:
            loss = (-1 * weight * log_probability).mean(axis=1)
        else:
            loss = (-1 * weight * labelss * log_probability).mean(axis=-1)

        return self.get_loss(loss)

class IOULoss(nn.Cell):
    """定义iou损失"""
    def __init__(self):
        super(IOULoss, self).__init__()
        self.abs = ops.Abs()

    def construct(self, pred, target):
        b = pred.shape[0] #batch
        IoU = 0
        for i in range(0,b):
            #compute the IoU of the foreground
            Iand1 = (target[i,:,:,:] * pred[i,:,:,:]).sum()
            Ior1 = (target[i,:,:,:]).sum() + (pred[i,:,:,:]).sum() - Iand1
            IoU1 = Iand1 / Ior1
            #IoU loss is (1-IoU1)
            IoU += (1-IoU1)

        return IoU / b


class MyHybridLoss(nn.Cell):
    '''
    IOULOSS + BCELOSS(0.64-0.01)+ focalloss(0.16-0.03）【样本不均衡】
    '''
    def __init__(self):
        super(MyHybridLoss, self).__init__()
        self.bceloss = nn.BCELoss(reduction='mean')
        self.iouloss = IOULoss()
        self.focalloss = MyFocalLoss()

    def construct(self, logits, label):
        total_loss = 0

        bce_loss = self.bceloss(logits, label)
        iou_loss = self.iouloss(logits, label)
        focal_loss = self.focalloss(logits, label)

        total_loss += bce_loss + iou_loss + focal_loss
        return total_loss




if __name__ == '__main__':
    # loss = TverskyLoss()
    loss = MyBCELoss()
    pred = Tensor(np.zeros((8,3,512,512)))
    preds = [pred, pred, pred, pred]
    target = Tensor(np.ones((8,3,512,512)))
    y = loss(preds,target)
    print(y,y.shape)
  