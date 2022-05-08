import torch.nn as nn
import torch.nn.functional as F


class Res_block(nn.Module):
    '''
    Res_block表示一个残差块
    如果输入输出通道相同，则是恒等映射h(x) = f(x) + x
    如果输入输出通道不同，则是线性映射,1*1的卷积
    '''

    def __init__(self, ch_in, ch_out, stride1, stride2) -> None:
        super(Res_block, self).__init__()
        self.blk = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=stride2, padding=1),
            nn.BatchNorm2d(ch_out)
        )
        self.extra = nn.Identity()
        # 输入输出通道数不同，线性映射
        if ch_in != ch_out:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=2, padding=0),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.blk(x) + self.extra(x))
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=1000) -> None:
        super(ResNet18, self).__init__()

        self.preconv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 由于stride决定维度，因此设置将stride作为Res_block参数输入
        # 参数意义: 输入通道，输出通道，残差块中第一层卷积步长，残差块中第二层卷积的步长
        self.block1 = Res_block(64, 64, 1, 1)
        self.block2 = Res_block(64, 64, 1, 1)
        self.block3 = Res_block(64, 128, 2, 1)
        self.block4 = Res_block(128, 128, 1, 1)
        self.block5 = Res_block(128, 256, 2, 1)
        self.block6 = Res_block(256, 256, 1, 1)
        self.block7 = Res_block(256, 512, 2, 1)
        self.block8 = Res_block(512, 512, 1, 1)
        # 池化操作
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # 全连接层
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # 7*7 conv + maxpool 输入 224*224*3 输出 64*56*56
        x = self.preconv(x)

        # 残差块1，输入 64*56*56 输出 64*56*56
        x = self.block1(x)
        # 残差块2 输入 64*56*56 输出 64*56*56
        x = self.block2(x)
        # 残差块3 输入 64*56*56 输出 128*28*28
        x = self.block3(x)
        # 残差块4 输入128*28*28 输出 128*28*28
        x = self.block4(x)
        # 残差块5 输入 128*28*28 输出 256*14*14
        x = self.block5(x)
        # 残差块6，输入 256*14*14 输出 256*14*14
        x = self.block6(x)
        # 残差块7，输入 256*14*14 输出 512*7*7
        x = self.block7(x)
        # 残差块8 输入 512*7*7 输出 512*7*7
        x = self.block8(x)
        # 平均池化 512*7*7-> 512*1*1
        x = self.avgpool(x)
        # Flatten 后面两维压平成一维
        x = x.view(x.size(0), -1)  # [512,1]
        # 全连接层 512,1 -> 1,2
        x = self.fc(x)
        x = F.softmax(x, dim=1)

        return x


def ResNet18_model(num_classes: int = 2):
    model = ResNet18(num_classes=num_classes)
    return model
