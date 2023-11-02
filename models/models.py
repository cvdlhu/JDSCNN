import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from lib.nn import SynchronizedBatchNorm2d
import math
from .attention_blocks import DualAttBlock
from .resnet import BasicBlock as ResBlock
from . import GSConv as gsc
import cv2
from .norm import Norm2d
from operators import PSPModule
from mynn import Norm2d, Upsample
from operators import conv_bn_relu, conv_sigmoid


class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def intersectionAndUnion(self, imPred, imLab, numClass):
        imPred = np.asarray(imPred.cpu()).copy()
        imLab = np.asarray(imLab.cpu()).copy()

        imPred += 1
        imLab += 1
        imPred = imPred * (imLab > 0)

        intersection = imPred * (imPred == imLab)
        (area_intersection, _) = np.histogram(
            intersection, bins=numClass, range=(1, numClass))

        (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
        (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
        area_union = area_pred + area_lab - area_intersection

        jaccard = area_intersection / area_union
        jaccard = (jaccard[1] + jaccard[2]) / 2
        return jaccard if jaccard <= 1 else 0

    def pixel_acc(self, pred, label, num_class):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 1).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)

        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)

        jaccard = []

        for i in range(1, num_class):
            v = (label == i).long()
            pred = (preds == i).long()
            anb = torch.sum(v * pred)
            try:
                j = anb.float() / (torch.sum(v).float() + torch.sum(pred).float() - anb.float() + 1e-10)
            except:
                j = 0

            j = j if j <= 1 else 0
            jaccard.append(j)

        return acc, jaccard

    def jaccard(self, pred, label):
        AnB = torch.sum(pred.long() & label)
        return AnB / (pred.view(-1).sum().float() + label.view(-1).sum().float() - AnB)


class SegmentationModule(SegmentationModuleBase):
    def __init__(self, crit, unet, num_class):
        super(SegmentationModule, self).__init__()
        self.crit = crit
        self.unet = unet
        self.num_class = num_class

    def forward(self, feed_dict, epoch, *, segSize=None):
        if segSize is None:
            p = self.unet(feed_dict['image'])
            loss = self.crit(p, feed_dict['mask'], epoch=epoch)
            acc = self.pixel_acc(torch.round(nn.functional.softmax(p[0], dim=1)).long(),
                                 feed_dict['mask'][0].long().cuda(), self.num_class)
            return loss, acc

        if segSize == True:
            p = self.unet(feed_dict['image'])
            pred = nn.functional.softmax(p[0], dim=1)
            return pred

        else:
            p = self.unet(feed_dict['image'])
            loss = self.crit((p[0], p[1]),
                             (feed_dict['mask'][0].long().unsqueeze(0), feed_dict['mask'][1].unsqueeze(0)))
            pred = nn.functional.softmax(p[0], dim=1)
            return pred, loss


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class ModelBuilder():
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def build_unet(self, num_class=1, arch='albunet', weights=''):
        arch = arch.upper()
        if arch == 'JDSCNN':
            unet = JDSCNN(num_classes=num_class)
        else:
            raise Exception('Architecture undefined!')

        if len(weights) > 0:
            unet.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
            print("Loaded pretrained UNet weights.")
        print('Loaded weights for unet')
        return unet


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                conv3x3_bn_relu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                conv3x3_bn_relu(in_channels, middle_channels),
                conv3x3_bn_relu(middle_channels, out_channels),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        return self.block(x)


class SkipConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        return self.block(x)


class JDSCNN(nn.Module):  # JDSCNN
    def __init__(self, num_classes=4, num_filters=32, pretrained=True, is_deconv=True):
        super(JDSCNN, self).__init__()

        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = torchvision.models.densenet121(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.d_in1 = conv_bn_relu(128, 128, 1, norm_layer=nn.BatchNorm2d)
        self.d_in2 = conv_bn_relu(256, 128, 1, norm_layer=nn.BatchNorm2d)
        self.d_in3 = conv_bn_relu(512, 128, 1, norm_layer=nn.BatchNorm2d)
        self.d_out1 = conv_bn_relu(128, 128, 1, norm_layer=nn.BatchNorm2d)
        self.d_out2 = conv_bn_relu(128, 256, 1, norm_layer=nn.BatchNorm2d)
        self.d_out3 = conv_bn_relu(128, 512, 1, norm_layer=nn.BatchNorm2d)
        self.gffgate1 = conv_sigmoid(128, 128)
        self.gffgate2 = conv_sigmoid(256, 128)
        self.gffgate3 = conv_sigmoid(512, 128)

        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(512, 1, kernel_size=1)
        self.c5 = nn.Conv2d(1024, 1, kernel_size=1)
        self.sc5 = nn.Conv2d(8, 512, kernel_size=1)
        self.cs5 = nn.Conv2d(1024, 512, kernel_size=1)

        self.d0 = nn.Conv2d(128, 64, kernel_size=1)
        self.res1 = ResBlock(64, 64)
        self.d1 = nn.Conv2d(64, 32, kernel_size=1)
        self.res2 = ResBlock(32, 32)
        self.d2 = nn.Conv2d(32, 16, kernel_size=1)
        self.res3 = ResBlock(16, 16)
        self.d3 = nn.Conv2d(16, 8, kernel_size=1)
        self.fuse = nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=False)

        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = gsc.GatedSpatialConv2d(32, 32)
        self.gate2 = gsc.GatedSpatialConv2d(16, 16)
        self.gate3 = gsc.GatedSpatialConv2d(8, 8)

        self.expand = nn.Sequential(nn.Conv2d(1, num_filters, kernel_size=1),
                                    Norm2d(num_filters),
                                    nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(self.encoder.features.conv0,
                                   self.encoder.features.norm0)
        self.conv2 = self.encoder.features.denseblock1
        self.conv2t = self.encoder.features.transition1
        self.conv3 = self.encoder.features.denseblock2
        self.conv3t = self.encoder.features.transition2
        self.conv4 = self.encoder.features.denseblock3
        self.conv4t = self.encoder.features.transition3
        self.conv5 = nn.Sequential(self.encoder.features.denseblock4,
                                   self.encoder.features.norm5)

        self.center = PSPModule(features=1024, norm_layer=nn.BatchNorm2d, out_features=512)
        norm_layer = nn.BatchNorm2d
        norm_layer = Norm2d

        self.dec5 = DualAttBlock(inchannels=[512, 1024], outchannels=512)
        self.dec4 = DualAttBlock(inchannels=[512, 512], outchannels=256)
        self.dec3 = DualAttBlock(inchannels=[512 + 256, 256], outchannels=128)
        self.dec2 = DualAttBlock(inchannels=[512 + 256 + 128, 128], outchannels=64)
        self.dec1 = DecoderBlock(512 + 256 + 128 + 64, 48, num_filters, is_deconv)
        self.dec0 = conv3x3_bn_relu(num_filters * 2, num_filters)
        self.final = nn.Conv2d(num_filters, self.num_classes, kernel_size=1)

    def forward(self, x, return_att=False):
        x_size = x.size()

        conv1 = self.conv1(x)
        conv2 = self.conv2t(self.conv2(conv1))
        conv3 = self.conv3t(self.conv3(conv2))
        conv4 = self.conv4t(self.conv4(conv3))
        conv5 = self.conv5(conv4)

        ss = F.interpolate(self.d0(conv2), x_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)
        c3 = F.interpolate(self.c3(conv3), x_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.d1(ss)
        ss, g1 = self.gate1(ss, c3)
        ss = self.res2(ss)
        ss = self.d2(ss)
        c4 = F.interpolate(self.c4(conv4), x_size[2:],
                           mode='bilinear', align_corners=True)
        ss, g2 = self.gate2(ss, c4)
        ss = self.res3(ss)
        ss = self.d3(ss)
        c5 = F.interpolate(self.c5(conv5), x_size[2:],
                           mode='bilinear', align_corners=True)
        ss, g3 = self.gate3(ss, c5)
        sc5 = F.interpolate(ss, size=[16, 16], mode='bilinear', align_corners=True)
        sc5 = self.sc5(sc5)
        ss = self.fuse(ss)
        ss = F.interpolate(ss, x_size[2:], mode='bilinear', align_corners=True)
        edge_out = self.sigmoid(ss)

        im_arr = np.mean(x.cpu().numpy(), axis=1).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i], 10, 100)
        canny = torch.from_numpy(canny).cuda().float()

        cat = torch.cat([edge_out, canny], dim=1)
        acts = self.cw(cat)
        acts = self.sigmoid(acts)
        edge = self.expand(acts)
        cs5 = self.cs5(conv5)
        conv5 = torch.cat([cs5, sc5], dim=1)

        conv2 = F.interpolate(conv2, scale_factor=2, mode='bilinear', align_corners=True)
        conv3 = F.interpolate(conv3, scale_factor=2, mode='bilinear', align_corners=True)
        conv4 = F.interpolate(conv4, scale_factor=2, mode='bilinear', align_corners=True)

        m2, m3, m4 = conv2, conv3, conv4
        m2_size = m2.size()[2:]
        m3_size = m3.size()[2:]
        m4_size = m4.size()[2:]

        g_m2 = self.gffgate1(m2)
        g_m3 = self.gffgate2(m3)
        g_m4 = self.gffgate3(m4)

        m2 = self.d_in1(m2)
        m3 = self.d_in2(m3)
        m4 = self.d_in3(m4)

        m2 = m2 + g_m2 * m2 + (1 - g_m2) * (Upsample(g_m3 * m3, size=m2_size) + Upsample(g_m4 * m4, size=m2_size))
        m3 = m3 + g_m3 * m3 + (1 - g_m3) * (
                Upsample(g_m2 * m2, size=m3_size) + Upsample(g_m4 * m4, size=m3_size))
        m4 = m4 + m4 * g_m4 + (1 - g_m4) * (
                Upsample(g_m3 * m3, size=m4_size) + Upsample(g_m2 * m2, size=m4_size))

        m2 = self.d_out1(m2)
        m3 = self.d_out2(m3)
        m4 = self.d_out3(m4)
        center = self.center(self.pool(conv5))
        dec5, _ = self.dec5([center, conv5])
        dec4, _ = self.dec4([dec5, m4])

        dec5_out = Upsample(dec5, size=(int(dec5.size()[2] * 2), int(dec5.size()[3] * 2)))
        dec3, _ = self.dec3([torch.cat([dec5_out, dec4], dim=1), m3])
        dec5_out = Upsample(dec5_out, size=(int(dec5_out.size()[2] * 2), int(dec5_out.size()[3] * 2)))
        dec4_out = Upsample(dec4, size=(int(dec4.size()[2] * 2), int(dec4.size()[3] * 2)))
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        dec2, _ = self.dec2([torch.cat([dec5_out, dec4_out, dec3], dim=1), m2])
        dec5_out = Upsample(dec5, size=(int(dec5.size()[2] * 8), int(dec5.size()[3] * 8)))
        dec4_out = Upsample(dec4, size=(int(dec4.size()[2] * 4), int(dec4.size()[3] * 4)))
        dec3_out = Upsample(dec3, size=(int(dec3.size()[2] * 2), int(dec3.size()[3] * 2)))
        dec1 = self.dec1(torch.cat([dec5_out, dec4_out, dec3_out, dec2], dim=1))  
        dec0 = self.dec0(torch.cat([dec1, edge], dim=1))

        x_out = self.final(dec0)

        return x_out, edge_out


