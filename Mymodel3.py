import torch
import os
import torch.nn.functional as F
from torch import nn
from torch.nn import init

class encoderunet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoderunet, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # ceil_mode参数取整的时候向上取整，该参数默认为False表示取整的时候向下取整
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        out = self.down_conv(x)
        out_pool = self.pool(out)
        return out, out_pool



class decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()
        # 反卷积
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_copy, x, interpolate=True):
        out = self.up(x)
        if interpolate:
            # 迭代代替填充， 取得更好的结果
            out = F.interpolate(out, size=(x_copy.size(2), x_copy.size(3)),
                                mode="bilinear", align_corners=True
                                )
        else:
            # 如果填充物体积大小不同
            diffY = x_copy.size()[2] - x.size()[2]
            diffX = x_copy.size()[3] - x.size()[3]
            out = F.pad(out, (diffX // 2, diffX - diffX // 2, diffY, diffY - diffY // 2))
        # 连接
        out = torch.cat([x_copy, out], dim=1)
        out_conv = self.up_conv(out)
        return out_conv



class BasicBlock(nn.Module):

    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class resnet(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=20):
        super(resnet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(8, 64, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  #3
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, img):
        out3 = F.relu(self.bn1(self.conv1(img)))
        out64 = self.layer1(out3)
        out32 = self.layer2(out64)
        return out64,out32



class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)##6,12,18
        self.bn_conv_3x3_1 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(out_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels)

        self.conv_1x1_3 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(out_channels)

    def forward(self, feature_map):
        feature_map_h = feature_map.size()[2]
        feature_map_w = feature_map.size()[3]

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))

        out_img = self.avg_pool(feature_map)

        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))
        out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear")

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1)
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))

        return out


class GET_Prototype(nn.Module):
    def __init__(self):
        super(GET_Prototype, self).__init__()

    def Weighted_P(self, supp_feat, mask):
        supp_feat = supp_feat * mask
        # print("supp_feat",supp_feat.shape)
        BB, CC, feat_h, feat_w = supp_feat.shape[0],supp_feat.shape[1],supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]

        supp_featmax = torch.max(supp_feat.reshape(BB, CC, -1),-1).values
        supp_featmax = supp_featmax.reshape(BB, CC, 1, 1)


        area2 = mask.sum(axis=[1,2,3])+0.001
        area2 = area2.reshape(BB,1,1,1)
        supp_feat2 = supp_feat.sum(axis=[2,3])
        supp_feat = supp_feat2.reshape(BB, CC, 1, 1)/area2


        return supp_feat,supp_featmax

    def forward(self, x, map):
        alln = map.size(1)
        _, H, W = x.size(0), x.size(2), x.size(3)

        prototype_meanlist = []
        prototype_maxlist = []


        for i in range(alln):
            map_slice = map[:, i, :, :].unsqueeze(1).float()

            map_slice = F.interpolate(map_slice, (H, W), mode='bilinear', align_corners=True)

            mask = map_slice

            prototypemean,prototypemax = self.Weighted_P(x, mask)

            prototypemean = prototypemean.squeeze(-1).squeeze(-1).unsqueeze(1)
            prototypemax= prototypemax.squeeze(-1).squeeze(-1).unsqueeze(1)

            prototype_meanlist.append(prototypemean)
            prototype_maxlist.append(prototypemax)

        prototype_blockmean = torch.cat(prototype_meanlist, dim=1)
        prototype_blockmax = torch.cat(prototype_maxlist, dim=1)

        return prototype_blockmean,prototype_blockmax

class MyModel(nn.Module):
    def __init__(self,num_classes, in_channels=10, freeze_bn=False, **_):
        super(MyModel, self).__init__()
        self.encoder = resnet()
        # self.aspp = ASPP(128, 128)
        self.propotype = GET_Prototype()

        self.theta = nn.Linear(128, 128)
        self.theta3 = nn.Linear(128, 128)
        self.theta2 = nn.Linear(1024, 1024)
        self.theta4 = nn.Linear(1024, 1024)
        self.down2 = encoderunet(128, 128)
        self.down3 = encoderunet(128, 256)
        self.down4 = encoderunet(256, 512)
        self.middle_conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up1 = decoder(1024, 512)
        self.up2 = decoder(512, 256)
        self.up3 = decoder(256, 128)
        self.up4 = decoder(128, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        self.final_conv2 = nn.Conv2d(20*2, num_classes, kernel_size=1)
        self.final_conv3 = nn.Conv2d(20 * 2, num_classes, kernel_size=1)
        self._initalize_weights()
        if freeze_bn:
            self.freeze_bn()
        in_channels = 20
        out_channels = 20
        self.up = nn.ConvTranspose2d(20, 20, kernel_size=2, stride=2)
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.upp = nn.ConvTranspose2d(20, 20, kernel_size=2, stride=2)
        self.upp_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )




    def _initalize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

        # self.phi = nn.Linear(self.in_channels, self.inter_channels)
        # self.g = nn.Linear(self.in_channels, self.inter_channels)

    def forward(self, x,xx1,xx2,xx3,xx1l,xx2l,xx3l):
        out64, out32 = self.encoder(x)
        # x = self.aspp(out32)   # torch.Size([5, 128, 32, 32])
        x = out32
        B, C, w, _ = x.size()
        x = x.view(B, C, -1)


        out164, out132 = self.encoder(xx1)
        # xx1 = self.aspp(out132)
        xx1 =out132
        x1_prototype1,x1_prototype1max = self.propotype(xx1, xx1l) #torch.Size([5, 12, 128])
        x1_prototype1 = self.theta(x1_prototype1)#torch.Size([5, 12, 128])
        x1_prototype1max = self.theta3(x1_prototype1max)
        attention1 = torch.matmul(x1_prototype1, x) #torch.Size([5, 12, 128])*torch.Size([5, 128, 32*32])=torch.Size([5, 12, 1024])
        attention11 = torch.matmul(x1_prototype1max, x)

        out264, out232 = self.encoder(xx2)
        # xx2 = self.aspp(out232)
        xx2 = out232
        x1_prototype2,x1_prototype2max = self.propotype(xx2, xx2l)  # torch.Size([5, 12, 128])
        x1_prototype2 = self.theta(x1_prototype2)  # torch.Size([5, 12, 128])
        x1_prototype2max = self.theta3(x1_prototype2max)
        attention2 = torch.matmul(x1_prototype2,x)  # torch.Size([5, 12, 128])*torch.Size([5, 128, 32*32])=torch.Size([5, 12, 1024])
        attention21 = torch.matmul(x1_prototype2max, x)

        out364, out332 = self.encoder(xx3)
        # xx3 = self.aspp(out332)
        xx3 = out332
        x1_prototype3,x1_prototype3max = self.propotype(xx3, xx3l)  # torch.Size([5, 12, 128])
        x1_prototype3 = self.theta(x1_prototype3)  # torch.Size([5, 12, 128])
        x1_prototype3max = self.theta3(x1_prototype3max)
        attention3 = torch.matmul(x1_prototype3,x)  # torch.Size([5, 12, 128])*torch.Size([5, 128, 32*32])=torch.Size([5, 12, 1024])
        attention31 = torch.matmul(x1_prototype3max, x)

        attention = attention1 + attention2 + attention3
        attention41 = attention11 + attention21 + attention31

        attention = self.theta2(attention)
        attention41 = self.theta4(attention41)

        attention = attention.view(B, 20, w, -1)
        attention41= attention41.view(B, 20, w, -1)

        attention = self.up(attention)
        attention =self.up_conv(attention)
        attention41 = self.upp(attention41)
        attention41 =self.upp_conv(attention41)


        # for x
        x2, x = self.down2(out32)
        x3, x = self.down3(x)

        x4, x = self.down4(x)
        x = self.middle_conv(x)
        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(out64, x)
        x = self.final_conv(x)
        attention = torch.cat([attention, attention41], 1)
        attention = self.final_conv3(attention)
        x = torch.cat([x, attention], 1)
        x = self.final_conv2(x)
        return x



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = MyModel(num_classes=20).cuda()
    x = torch.rand(1, 8, 64, 64).cuda()
    x1 = torch.rand(1, 8, 64, 64).cuda()
    x2 = torch.rand(1, 8, 64, 64).cuda()
    x3 = torch.rand(1, 8, 64, 64).cuda()
    x1l= torch.rand(1, 20, 64, 64).cuda()
    x2l = torch.rand(1, 20, 64, 64).cuda()
    x3l = torch.rand(1, 20, 64, 64).cuda()
    out = model(x,x1,x2,x3,x1l,x2l,x3l)

    from thop import profile
    flops, params = profile(model, inputs=(x,x1,x2,x3,x1l,x2l,x3l,))
    print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))
    print(" %.2f | %.2f" % (params / (1000 ** 2), flops / (1000 ** 3)))