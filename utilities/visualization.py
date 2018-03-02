import torch
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

MEAN = np.array([0.03072981, 0.03072981, 0.01682784])
STD = np.array([0.17293351, 0.12542403, 0.0771413 ])

def show_mask(model, img_path, mask_path=None, image_size=512, threshold=0.5, show_img=True, mean=MEAN, std=STD):
    mask_cmap = colors.ListedColormap(['black', '#bd0b42'])
    mask_cmap = mask_cmap(np.arange(2))
    mask_cmap[:,-1] = np.linspace(0, 1, 2)
    mask_cmap = colors.ListedColormap(mask_cmap)
    pred_cmap = colors.ListedColormap(['black', '#42f49e'])
    pred_cmap = pred_cmap(np.arange(2))
    pred_cmap[:,-1] = np.linspace(0, 1, 2)
    pred_cmap = colors.ListedColormap(pred_cmap)
    plt.figure(figsize=(10,10))
    plt.rcParams['axes.facecolor']='black'
    plt.xticks([])
    plt.yticks([])
    plt.title(img_path)
    
    img1 = cv2.imread(img_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    size = (img1.shape[1], img1.shape[0])
    img = cv2.resize(img1, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean, std)(img)
    img = img.unsqueeze(0)
    img = Variable(img.cuda())
    model = model.cuda()
    pred_mask = model(img)
    pred_mask = F.sigmoid(pred_mask)
    pred_mask = (pred_mask > threshold) if threshold is not None else pred_mask
    pred_mask = np.squeeze((pred_mask.data).cpu().numpy())
    
    if show_img:
        plt.imshow(img1)    
    if mask_path is not None:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)
        plt.imshow(mask, cmap=mask_cmap, alpha=(.6 if show_img else 1))
    if pred_mask is not None:
        pred_mask = cv2.resize(pred_mask, size, interpolation=cv2.INTER_LINEAR)
        plt.imshow(pred_mask, cmap=pred_cmap, alpha=.6)