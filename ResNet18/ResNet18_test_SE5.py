import cv2
import time
import argparse
import numpy as np
from PIL import Image
import sophon.sail as sail


def preprocess(img):
    image = Image.fromarray(img)
    # 训练时预处理函数transforms.Resize()默认resample=2, 此处image.reseize()默认resample=3, resample参数需手动加上
    resized_img = np.array(image.resize((224, 224), resample=2))
    out = np.array(resized_img / 255., dtype=np.float32)
    return out.transpose((2, 0, 1))


def main():
    start_time = time.time()
    img = cv2.imread(ARGS.img_path)
    img_array = preprocess(img)

    # sail core (inference)
    net = sail.Engine(ARGS.bmodel, ARGS.tpu_id, sail.IOMode.SYSIO)
    graph_name = net.get_graph_names()[0]  # get net name
    input_names = net.get_input_names(graph_name)  # [input_names], 输入节点名列表
    output_names = net.get_output_names(graph_name)  # [output_names],输出节点名列表
    input_data = {input_names[0]: np.expand_dims(img_array, axis=0)}

    # 模型推理，推理结果为字典，字典的key值分别是输出节点名
    prediction = net.process(graph_name, input_data)
    end_time = time.time()
    timer = end_time - start_time
    print("-" * 66)
    print('The probability of CATS: %.5f' % prediction[output_names[0]][:, 0])
    print('The probability of DOGS: %.5f' % prediction[output_names[0]][:, 1])
    print("Time consuming: %.5f sec" % timer)
    print("-" * 66)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='for sail py test')
    PARSER.add_argument('--bmodel', default='./ResNet18_bmodel/compilation.bmodel')
    PARSER.add_argument('--img_path', default='./images/cat.10000.jpg')
    PARSER.add_argument('--tpu_id', default=0, type=int, required=False)
    ARGS = PARSER.parse_args()
    main()
