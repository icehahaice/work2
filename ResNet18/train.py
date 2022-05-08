import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from resnet18 import ResNet18_model as ResNet_model
import DogCatDataset


def main():
    # Step 0:查看torch版本、设置device
    print('Pytorch Version = ', torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1:准备数据集
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_data = DogCatDataset.DogCatDataset(root_path=os.path.join(os.getcwd(), 'data/train_set'),
                                             transform=train_transform)
    train_dataloader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

    # Step 2: 初始化模型
    model = ResNet_model(num_classes=2)
    # trained_weight = torch.load('./logs/resnet18_Cat_Dog_ep015-loss1.001.pth')
    # model.load_state_dict(trained_weight)
    model.to(device)

    # Step 3:交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    # Step 4:选择优化器
    LR = 0.001
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    # Step 5:设置学习率下降策略
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

    # Step 6:训练网络
    model.train()
    MAX_EPOCH = 5  # 设置epoch=20

    for epoch in range(MAX_EPOCH):
        loss_total = 0
        total_sample = 0
        accuracy_total = 0
        # desc=f'Epoch {epoch + 1}/{MAX_EPOCH}'
        with tqdm(train_dataloader, postfix=dict, mininterval=0.3) as pbar:
            for iteration, data in enumerate(train_dataloader):
                img, label = data
                img, label = img.to(device), label.to(device)
                output = model(img)

                optimizer.zero_grad()
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                _, predicted_label = torch.max(output, 1)

                total_sample += label.size(0)
                accuracy_total += torch.mean((predicted_label == label).type(torch.FloatTensor)).item()
                loss_total += loss.item()

                pbar.set_postfix(**{'total_loss': loss_total / (iteration + 1),
                                    'accuracy': accuracy_total / (iteration + 1),
                                    'lr': optimizer.param_groups[0]['lr']})
                pbar.update(1)

        scheduler.step()  # 更新学习率

        # Step 7: 存储权重
        # torch.save(model.state_dict(), 'C:/Users/hao/Desktop/ResNet18/logs/resnet18_Cat_Dog_ep%03d-loss%.3f.pth' %
        #            ((epoch + 1), loss_total / iteration + 1))

    print('train finish!')
    traced_module = torch.jit.trace(model.cpu(), torch.rand(1, 3, 224, 224))
    # torch.jit.save(traced_module, 'C:/Users/hao/Desktop/ResNet18/logs/resnet18_trace.zip')
    model.to(device)

if __name__ == '__main__':
    main()
