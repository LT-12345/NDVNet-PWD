import h5py
import numpy as np
import imageio
from scipy.ndimage import zoom, rotate
from skimage.transform import resize
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import imageio
from scipy.ndimage import gaussian_filter
import torch.nn as nn
import torch.nn.init as init
import os
import pandas as pd

# Parameters
height = 256  # 输入模型的图像尺寸
width = 256  # 输入模型的图像尺寸
channels = 3  # RGB通道数
ndvi_evi_channels = 2  # NDVI和EVI通道数
train_number =594   # 随机分配用于生成训练集的图像数量
val_number = 170  # 随机分配用于生成验证集的图像数量
test_number = 84  # 随机分配用于生成测试集的图像数量
all = train_number + val_number + test_number

Image.MAX_IMAGE_PIXELS = None  # 或者设置为一个较大的值，比如1000000000

# 加载NDVI和EVI数据
def load_ndvi_evi_data():
    ndvi_evi_data = {}
    df = pd.read_csv('ndvi_evi_results.csv')
    for _, row in df.iterrows():
        filename = row['filename']
        ndvi_evi_data[filename] = {
            'ndvi_mean': row['ndvi_mean'],
            'evi_mean': row['evi_mean']
        }
    return ndvi_evi_data

# 准备数据集
Tr_list = glob.glob("images/*.png")  # 图像存储文件夹
Data_train_2018 = np.zeros([all, height, width, channels])
NDVI_EVI_train_2018 = np.zeros([all, height, width, ndvi_evi_channels])
Label_train_2018 = np.zeros([all, height, width])

print('Reading')
print(len(Tr_list))

# 加载NDVI和EVI数据
ndvi_evi_data = load_ndvi_evi_data()

# 随机打乱数据集索引
random.shuffle(Tr_list)

# 随机旋转和翻转
def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label
 # 随机角度旋转
def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = rotate(image, angle, reshape=False, order=3)
    label = rotate(label, angle, reshape=False, order=0)
    return image, label


class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.sig = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b * self.groups, -1, h, w)  # bs*g,dim//g,h,w
        xn = x * self.avg_pool(x)  # bs*g,dim//g,h,w
        xn = xn.sum(dim=1, keepdim=True)  # bs*g,1,h,w

        t = xn.view(b * self.groups, -1)  # bs*g,h*w

        t = t - t.mean(dim=1, keepdim=True)  # bs*g,h*w
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std  # bs*g,h*w
        t = t.view(b, self.groups, h, w)  # bs,g,h*w

        t = t * self.weight + self.bias  # bs,g,h*w
        t = t.view(b * self.groups, 1, h, w)  # bs*g,1,h*w
        x = x * self.sig(t)
        x = x.view(b, c, h, w)

        return x

# 数据增强函数
def data_augmentation(image, label):
    # 随机旋转和翻转图像和标签
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    
    # 添加亮度调整 - 仅应用于图像，不应用于标签
    if np.random.random() > 0.5:
        # 亮度因子：0.7-1.3之间的随机值
        brightness_factor = np.random.uniform(0.7, 1.3)
        image = image * brightness_factor
        # 确保像素值在有效范围内
        image = np.clip(image, 0, 1) if image.max() <= 1 else np.clip(image, 0, 255)
    
    # 添加对比度调整 - 仅应用于图像，不应用于标签
    if np.random.random() > 0.5:
        # 对比度因子：0.7-1.3之间的随机值
        contrast_factor = np.random.uniform(0.7, 1.3)
        mean = np.mean(image, axis=(0, 1))
        image = (image - mean) * contrast_factor + mean
        # 确保像素值在有效范围内
        image = np.clip(image, 0, 1) if image.max() <= 1 else np.clip(image, 0, 255)
    
    return image, label

for idx in range(len(Tr_list)):
    print(idx + 1)
    img = imageio.v2.imread(Tr_list[idx])
    img_resized = resize(img, (height, width, channels), mode='reflect', anti_aliasing=True)
    
    b = Tr_list[idx]
    b = b[-8:-4]  # 更简洁的字符串切片方式
    add = f"masks/{b}.png"  # 使用f-string进行字符串格式化
    img2 = imageio.v2.imread(add)
    img2_resized = resize(img2, (height, width), mode='reflect', preserve_range=True)
    
    # 获取对应的NDVI和EVI数据
    filename = os.path.basename(Tr_list[idx])
    if filename in ndvi_evi_data:
        ndvi_mean = ndvi_evi_data[filename]['ndvi_mean']
        evi_mean = ndvi_evi_data[filename]['evi_mean']
    else:
        ndvi_mean = 0
        evi_mean = 0
    
    # 创建NDVI和EVI通道
    ndvi_channel = np.full((height, width), ndvi_mean)
    evi_channel = np.full((height, width), evi_mean)
    ndvi_evi = np.stack([ndvi_channel, evi_channel], axis=-1)
    
    # 应用数据增强
    augmented_sample = data_augmentation(img_resized, img2_resized)
    
    # 存储数据
    Data_train_2018[idx, :, :, :] = augmented_sample[0]
    NDVI_EVI_train_2018[idx, :, :, :] = ndvi_evi
    Label_train_2018[idx, :, :] = augmented_sample[1]

print('Reading your dataset finished')

# 制作训练、验证和测试集
Train_img = Data_train_2018[0:train_number, :, :, :]
Train_ndvi_evi = NDVI_EVI_train_2018[0:train_number, :, :, :]
Validation_img = Data_train_2018[train_number:train_number + val_number, :, :, :]
Validation_ndvi_evi = NDVI_EVI_train_2018[train_number:train_number + val_number, :, :, :]
Test_img = Data_train_2018[train_number + val_number:all, :, :, :]
Test_ndvi_evi = NDVI_EVI_train_2018[train_number + val_number:all, :, :, :]

Train_mask = Label_train_2018[0:train_number, :, :]
Validation_mask = Label_train_2018[train_number:train_number + val_number, :, :]
Test_mask = Label_train_2018[train_number + val_number:all, :, :]

# 保存数据集
np.save('data_train.npy', Train_img)
np.save('ndvi_evi_train.npy', Train_ndvi_evi)
np.save('data_test.npy', Test_img)
np.save('ndvi_evi_test.npy', Test_ndvi_evi)
np.save('data_val.npy', Validation_img)
np.save('ndvi_evi_val.npy', Validation_ndvi_evi)
np.save('mask_train.npy', Train_mask)
np.save('mask_test.npy', Test_mask)
np.save('mask_val.npy', Validation_mask)


