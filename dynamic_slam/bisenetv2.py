"""
BiSeNetV2 Semantic Segmentation Module
Paper Section 3.1: 基于BiSeNetV2的语义分割

Used by ORB-SLAM2 to detect dynamic objects (people) in RGB images.
Outputs a binary mask where 1=dynamic (person), 0=static (background).

The paper: "由于主要动态目标为人物，因此语义分割图仅针对人物进行"
"""

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- BiSeNetV2 Network (paper Section 3.1.2, Figure 3.3) ----

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride,
                              padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DetailBranch(nn.Module):
    """Detail Branch: wide channels, shallow, preserves spatial detail."""
    def __init__(self):
        super().__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(3, 64, ks=3, stride=2, padding=1),
            ConvBNReLU(64, 64, ks=3, stride=1, padding=1),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(64, 64, ks=3, stride=2, padding=1),
            ConvBNReLU(64, 64, ks=3, stride=1, padding=1),
            ConvBNReLU(64, 64, ks=3, stride=1, padding=1),
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(64, 128, ks=3, stride=2, padding=1),
            ConvBNReLU(128, 128, ks=3, stride=1, padding=1),
            ConvBNReLU(128, 128, ks=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.S3(self.S2(self.S1(x)))


class StemBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvBNReLU(3, 16, ks=3, stride=2, padding=1)
        self.left = nn.Sequential(
            ConvBNReLU(16, 8, ks=1, stride=1, padding=0),
            ConvBNReLU(8, 16, ks=3, stride=2, padding=1),
        )
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fuse = ConvBNReLU(32, 16, ks=3, stride=1, padding=1)

    def forward(self, x):
        feat = self.conv(x)
        return self.fuse(torch.cat([self.left(feat), self.right(feat)], dim=1))


class GELayerS1(nn.Module):
    """Gather-and-Expansion Layer, stride=1."""
    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super().__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, ks=3, stride=1,
                                padding=1, groups=in_chan)
        self.conv2 = nn.Conv2d(in_chan, mid_chan, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_chan)
        self.conv3 = nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = (in_chan == out_chan)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.relu(self.bn2(self.conv2(feat)))
        feat = self.bn3(self.conv3(feat))
        if self.shortcut:
            feat = feat + x
        return self.relu(feat)


class GELayerS2(nn.Module):
    """Gather-and-Expansion Layer, stride=2."""
    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super().__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1_dw = ConvBNReLU(in_chan, in_chan, ks=3, stride=2,
                                   padding=1, groups=in_chan)
        self.conv1_pw = nn.Conv2d(in_chan, mid_chan, kernel_size=1, bias=False)
        self.bn1_pw = nn.BatchNorm2d(mid_chan)
        self.conv2_dw = ConvBNReLU(mid_chan, mid_chan, ks=3, stride=2,
                                   padding=1, groups=mid_chan)
        self.conv2_pw = nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=False)
        self.bn2_pw = nn.BatchNorm2d(out_chan)
        self.shortcut_conv = nn.Conv2d(in_chan, out_chan, kernel_size=3,
                                       stride=2, padding=1, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1_dw(x)
        feat = self.relu(self.bn1_pw(self.conv1_pw(feat)))
        feat = self.conv2_dw(feat)
        feat = self.bn2_pw(self.conv2_pw(feat))
        shortcut = self.shortcut_bn(self.shortcut_conv(x))
        return self.relu(feat + shortcut)


class SemanticBranch(nn.Module):
    """Semantic Branch: deep, narrow, captures high-level context."""
    def __init__(self):
        super().__init__()
        self.stem = StemBlock()
        self.stage3 = nn.Sequential(
            GELayerS2(16, 32, exp_ratio=6),
            GELayerS1(32, 32, exp_ratio=6),
        )
        self.stage4 = nn.Sequential(
            GELayerS2(32, 64, exp_ratio=6),
            GELayerS1(64, 64, exp_ratio=6),
        )
        self.stage5 = nn.Sequential(
            GELayerS2(64, 128, exp_ratio=6),
            GELayerS1(128, 128, exp_ratio=6),
            GELayerS1(128, 128, exp_ratio=6),
            GELayerS1(128, 128, exp_ratio=6),
        )

    def forward(self, x):
        return self.stage5(self.stage4(self.stage3(self.stem(x))))


class BGA(nn.Module):
    """Bilateral Guided Aggregation layer."""
    def __init__(self, detail_chan=128, semantic_chan=128, out_chan=128):
        super().__init__()
        self.detail_branch = nn.Sequential(
            ConvBNReLU(detail_chan, out_chan, ks=3, stride=1, padding=1),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=2,
                     padding=1, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.semantic_branch = nn.Sequential(
            ConvBNReLU(semantic_chan, out_chan, ks=3, stride=1, padding=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=1,
                     padding=1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
        )
        self.fuse = ConvBNReLU(out_chan, out_chan, ks=3, stride=1, padding=1)

    def forward(self, detail, semantic):
        d_feat = self.detail_branch(detail)
        s_feat = self.semantic_branch(semantic)
        s_feat = F.interpolate(s_feat, size=d_feat.shape[2:],
                               mode='bilinear', align_corners=False)
        return self.fuse(d_feat * torch.sigmoid(s_feat))


class BiSeNetV2(nn.Module):
    """BiSeNetV2 for person segmentation in dynamic SLAM."""

    def __init__(self, num_classes=21):  # PASCAL VOC: 20 + background
        super().__init__()
        self.detail_branch = DetailBranch()
        self.semantic_branch = SemanticBranch()
        self.bga = BGA(detail_chan=128, semantic_chan=128, out_chan=128)
        self.seg_head = nn.Sequential(
            ConvBNReLU(128, 128, ks=3, stride=1, padding=1),
            nn.Conv2d(128, num_classes, kernel_size=1, bias=False),
        )
        self.dynamic_classes = {14}  # PASCAL VOC: person=14

    def forward(self, x):
        detail = self.detail_branch(x)
        semantic = self.semantic_branch(x)
        fused = self.bga(detail, semantic)
        logits = self.seg_head(fused)
        return F.interpolate(logits, size=x.shape[2:],
                             mode='bilinear', align_corners=False)


# ---- Inference Interface ----

class SemanticSegmenter:
    """
    Dynamic object segmenter called from ORB-SLAM2 tracking thread.

    Usage from C++ (via subprocess or embedded Python):
        segmenter = SemanticSegmenter(model_path)
        mask = segmenter.segment(rgb_image)  # returns [H,W] uint8, 1=person
    """

    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_path and os.path.exists(model_path):
            self.model = BiSeNetV2(num_classes=21)
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device), strict=False)
            self.model_type = 'bisenetv2'
        else:
            # Use torchvision DeepLabV3 as pretrained alternative
            from torchvision.models.segmentation import (
                deeplabv3_mobilenet_v3_large,
                DeepLabV3_MobileNet_V3_Large_Weights
            )
            weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
            self.model = deeplabv3_mobilenet_v3_large(weights=weights)
            self.model_type = 'deeplabv3'

        self.model.to(self.device)
        self.model.eval()

    def segment(self, rgb_image):
        """
        Args:
            rgb_image: numpy array [H, W, 3] RGB, uint8

        Returns:
            mask: numpy array [H, W] uint8, 1=dynamic(person), 0=static
        """
        H, W = rgb_image.shape[:2]
        img_tensor = torch.from_numpy(rgb_image).float().permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img_tensor = ((img_tensor - mean) / std).to(self.device)

        with torch.no_grad():
            if self.model_type == 'bisenetv2':
                logits = self.model(img_tensor)
                probs = F.softmax(logits, dim=1)
                person_prob = probs[:, 14:15, :, :]  # PASCAL VOC person=14
            else:
                output = self.model(img_tensor)['out']
                probs = F.softmax(output, dim=1)
                person_prob = probs[:, 15:16, :, :]  # COCO person=15

            mask = (person_prob > 0.5).float()
            mask = F.interpolate(mask, size=(H, W), mode='nearest')
            return mask.squeeze().cpu().numpy().astype(np.uint8)
