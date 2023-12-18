import numpy as np
import cv2
from config import cfg


def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


def rotate(xb, yb, angle):
    size = xb.shape[1]
    M_rotate = cv2.getRotationMatrix2D(
        (size / 2, size / 2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (size, size))
    yb = cv2.warpAffine(yb, M_rotate, (size, size))
    return xb, yb


def blur(img):
    img = cv2.blur(img, (3, 3))
    return img


def add_noise(img):
    for i in range(200):  # 添加点噪声
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img

# 数据增强函数xb为image yb为label
def data_augment(xb, yb):
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 90)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 180)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转cv2.flip
        yb = cv2.flip(yb, 1)
    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb, 1.0)
    if np.random.random() < 0.25:
        xb = blur(xb)
    if np.random.random() < 0.2:
        xb = add_noise(xb)
    return xb, yb


def printDataset(dataset_list, name_list):
    """显示数据集"""
    dataset_sizes = []
    for dataset in dataset_list:
        dataset_sizes.append(dataset.get_dataset_size())
    row = len(dataset_list)      # 画布行数
    column = max(dataset_sizes)  # 画布列数
    pos = 1
    for i in range(row):
        for data in dataset_list[i].create_dict_iterator(output_numpy=True):
            plt.subplot(row, column, pos)                          # 显示位置
            plt.imshow(data['image'].squeeze(), cmap=plt.cm.gray)  # 显示内容
            plt.title(data['label'])                               # 显示标题
            print(name_list[i], " shape:", data['image'].shape, "label:", data['label'])
            pos = pos + 1
        pos = column * (i + 1) + 1

