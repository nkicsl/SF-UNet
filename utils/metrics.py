import numpy as np
import os
import torch
from PIL import Image
import monai

Dice = monai.metrics.DiceMetric(ignore_empty=False)

def mask2onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

def onehot2mask(mask):
    """
    Converts a mask (K, H, W) to (H,W)
    """
    _mask = np.argmax(mask, axis=0).astype(np.uint8)
    return _mask


def fast_hist(a, b, n):
    """
    Args:
        a: a是转化成一维数组的标签，形状(H×W,)
        b: b是转化成一维数组的预测值，形状(H×W,)
        n: num_classes

    Returns: 混沌矩阵，写对角线上的为分类正确的像素点

    """
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    """

    Args:
        hist: 输入的混淆矩阵

    Returns: 每一类iou值

    """
    # 矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,)
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def average_ac(hist):
    """

    Args:
        hist: 输入的混淆矩阵

    Returns: 像素预测的平均准确率

    """
    return np.sum(np.diag(hist)) / np.sum(hist)


def per_class_ac(hist):
    """

    Args:
        hist: 输入的混淆矩阵

    Returns:每一类别像素预测的准确率

    """
    return np.diag(hist) / (hist.sum(0))


def per_class_dice(hist):
    """

    Args:
        hist: 输入的混淆矩阵

    Returns: 每一类别像素预测的准确率

    """
    iou_classes = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    dice_classes = []
    for i in range(len(iou_classes)):
        dice = 2 * iou_classes[i] / (1 + iou_classes[i])
        dice_classes.append(dice)
    return dice_classes


def miou(iou_list):
    mean_iou = sum(iou_list) / len((iou_list))
    return mean_iou


def Mpa(pa_list):
    mpa = sum(pa_list) / len(pa_list)
    return mpa


def Mdice(dice_lsit):
    mdice = sum(dice_lsit) / len(dice_lsit)
    return mdice


def test_in_train(img_dir, label_dir, n_classes, b=1):
    test_dir =img_dir
    test_list = os.listdir(test_dir)
    test_list.sort()

    label_dir = label_dir
    label_list = os.listdir(label_dir)
    label_list.sort()

    n = n_classes
    iou_list = torch.zeros((b,n_classes))
    dice_list = torch.zeros((b,n_classes))

    for i in range(len(test_list)):
        img = Image.open(os.path.join(test_dir, test_list[i]))
        image_data = np.array(img)
        image_data[image_data == 255] = 1
        image_onehot = torch.tensor(mask2onehot(image_data, n)).unsqueeze(0)


        label = Image.open(os.path.join(label_dir, label_list[i]))
        label_data = np.array(label)
        label_data[label_data == 255] = 1
        label_onehot = torch.tensor(mask2onehot(label_data, n)).unsqueeze(0)

        iou = monai.metrics.compute_iou(image_onehot, label_onehot)
        dice = Dice(image_onehot, label_onehot)

        dice_list += dice
        iou_list += iou


    return iou_list[:, 1] / len(test_list), dice_list[:, 1] / len(test_list)

def test_in_train_ACDC(img_dir, label_dir, n_classes, b=1):
    test_dir =img_dir
    test_list = os.listdir(test_dir)
    test_list.sort()

    label_dir = label_dir
    label_list = os.listdir(label_dir)
    label_list.sort()

    n = n_classes
    iou_list = torch.zeros((b,n_classes))
    dice_list = torch.zeros((b,n_classes))

    for i in range(len(test_list)):
        img = Image.open(os.path.join(test_dir, test_list[i]))
        image_data = np.array(img)
        image_data[image_data == 255] = 1
        image_onehot = torch.tensor(mask2onehot(image_data, n)).unsqueeze(0)


        label = Image.open(os.path.join(label_dir, label_list[i]))
        label_data = np.array(label)
        label_data[label_data == 255] = 1
        label_onehot = torch.tensor(mask2onehot(label_data, n)).unsqueeze(0)

        iou = monai.metrics.compute_iou(image_onehot, label_onehot, ignore_empty=False)
        dice = Dice(image_onehot, label_onehot)

        dice_list += dice
        iou_list += iou


    return iou_list[:, 1:4] / len(test_list), dice_list[:, 1:4] / len(test_list)