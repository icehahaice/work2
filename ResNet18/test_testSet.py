import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import DogCatDataset


def main():
    # Step 0:查看torch版本、设置device
    print('Pytorch Version = ', torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1:准备数据集
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    test_data = DogCatDataset.DogCatDataset(root_path=os.path.join(os.getcwd(), 'data/test_set'),
                                            transform=test_transform)
    test_dataloader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    # Step 2: 初始化网络
    model = models.resnet18()

    # Step 3：加载训练好的权重
    trained_weight = torch.load('./resnet18_Cat_Dog.pth')
    model.load_state_dict(trained_weight)
    model.to(device)

    # Steo 4：网络推理
    model.eval()

    accuracy_total = 0
    with torch.no_grad():
        for batch_nums, data in enumerate(test_dataloader):
            img, label = data
            img = img.to(device)
            label = label.to(device)
            output = model(img)

            _, predicted_label = torch.max(output, 1)

            accuracy_total += torch.mean((predicted_label == label).type(torch.FloatTensor)).item()

    # Step 5:打印分类准确率
    print('test accuracy is ', accuracy_total / batch_nums)


if __name__ == '__main__':
    main()
