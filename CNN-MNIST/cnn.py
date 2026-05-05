import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
class simplecnn(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1),#经过此卷积，图像大小不变
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),#池化降低参数量
            nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)#通道大小32，图像大小56
        )
        self.classifier=nn.Sequential(
            nn.Linear(32*56*56,128),#全连接层作特征提取
            nn.ReLU(),
            nn.Linear(128,num_classes)#num_classes是分类
        )
    def forward(self,x):
        x=self.features(x)#将图像输入到特征提取层进行特征提取
        x=x.view(x.size(0),-1)#展开为1维，x.size(0)，batch_size=64，-1=32*56*56
        x=self.classifier(x)
        return x
num_classes=10
model=simplecnn(num_classes)
input=torch.randn(64,3,224,224)
output=model(input)
print(output.shape)