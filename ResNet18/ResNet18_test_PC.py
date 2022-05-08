# coding:utf-8
import cv2
import time
import torch
import argparse
from torchvision import transforms


def preprocess(image):
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img_tensor = test_transform(image).unsqueeze(0)
    return img_tensor


def main():
    start_time = time.time()
    img = cv2.imread(ARGS.img_path)
    img_tensor = preprocess(img)

    model = torch.jit.load(ARGS.model_path)

    # 模型推理
    prediction = model(img_tensor)
    end_time = time.time()
    timer = end_time - start_time
    print("-----------------------------------")
    print('The probability of CATS: %.5f' % prediction[:, 0])
    print('The probability of DOGS: %.5f' % prediction[:, 1])
    print("Time consuming: %.5f sec" % timer)
    print("-----------------------------------")


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='for PC py test')
    PARSER.add_argument('--img_path', default='./data/test_set/dog.10000.jpg')
    PARSER.add_argument('--model_path', default='./logs/resnet18_traced.zip')
    ARGS = PARSER.parse_args()
    main()
