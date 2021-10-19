from collections import namedtuple

import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

vgg_features = models.vgg16(pretrained=True).features

feature1 = vgg_features[:3]
feature2 = vgg_features[3:8]
feature3 = vgg_features[8:13]
feature4 = vgg_features[13:20]
feature5 = vgg_features[20:27]

gt_image = torch.randn([32, 3,128,128])
pred_image = torch.randn([32, 3,128,128])

names= ['f1', 'f2', 'f3', 'f4', 'f5']

h = feature1(gt_image)
h_conv1_2 = h
h = feature2(h)
h_conv2_2 = h
h = feature3(h)
h_relu3_2 = h
h = feature4(h)
h_relu4_2 = h
h = feature5(h)
h_relu5_2 = h
vgg_outputs = namedtuple("VggOutputs", names)

out_gt = vgg_outputs(h_conv1_2, h_conv2_2, h_relu3_2, h_relu4_2, h_relu5_2)

h = feature1(pred_image)
h_conv1_2 = h
h = feature2(h)
h_conv2_2 = h
h = feature3(h)
h_relu3_2 = h
h = feature4(h)
h_relu4_2 = h
h = feature5(h)
h_relu5_2 = h
vgg_outputs = namedtuple("VggOutputs", names)
out_pred = vgg_outputs(h_conv1_2, h_conv2_2, h_relu3_2, h_relu4_2, h_relu5_2)

feature_gt, feature_pred = [gt_image], [pred_image]

for k in names:
    feature_gt.append(getattr(out_gt, k))
    feature_pred.append(getattr(out_pred, k))

# print(feature_gt.shape)

losses = []
for i in range(len(feature_gt)):
    l = F.mse_loss(feature_gt[i], feature_pred[i], reduction='none')
    l = torch.mean(l)
    # print(l, torch.mean(l))
    losses.append(l)

# print(losses[0].shape)
vgg_losses = [x.item() for x in losses]
loss = torch.stack(losses).sum()

print(loss, vgg_losses)