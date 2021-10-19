from collections import namedtuple

import torch
from torchvision import models

class Perceptual_loss(nn.Module):

    def __init__(self, requires_grad=False):
        super(Perceptual_loss, self).__init__()
        self.vgg_features = models.vgg16(pretrained=True).features
        self.feature1 = vgg_features[:3]
        self.feature2 = vgg_features[3:8]
        self.feature3 = vgg_features[8:13]
        self.feature4 = vgg_features[13:20]
        self.feature5 = vgg_features[20:27]
        self.names= ['f1', 'f2', 'f3', 'f4', 'f5']

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False


        def forward(self, gt_image, pred_image, mask=None):
            h = self.feature1(gt_image)
            h_conv1_2 = h
            h = self.feature2(h)
            h_conv2_2 = h
            h = self.feature3(h)
            h_relu3_2 = h
            h = self.feature4(h)
            h_relu4_2 = h
            h = self.feature5(h)
            h_relu5_2 = h
            vgg_outputs = namedtuple("VggOutputs", self.names)
            out_gt = vgg_outputs(h_conv1_2, h_conv2_2, h_relu3_2, h_relu4_2, h_relu5_2)

            h = self.feature1(pred_image)
            h_conv1_2 = h
            h = self.feature2(h)
            h_conv2_2 = h
            h = self.feature3(h)
            h_relu3_2 = h
            h = self.feature4(h)
            h_relu4_2 = h
            h = self.feature5(h)
            h_relu5_2 = h
            vgg_outputs = namedtuple("VggOutputs", self.names)
            out_pred = vgg_outputs(h_conv1_2, h_conv2_2, h_relu3_2, h_relu4_2, h_relu5_2)

            feature_gt, feature_pred = [gt_image, pred_image]

            for k in self.names:
                feature_gt.append(getattr(out_gt, k))
                feature_pred.append(getattr(out_pred, k))

            losses = []
            for i in range(len(feature_gt)):
                l = F.mse_loss(feature_gt[i], feature_pred[i], reduction='none')
                l = torch.mean(l)
                # print(l, torch.mean(l))
                losses.append(l)

            # print(losses[0].shape)
            vgg_losses = [x.item() for x in losses]
            loss = torch.stack(losses).sum()

            # print(loss, vgg_losses)

            return loss, vgg_losses