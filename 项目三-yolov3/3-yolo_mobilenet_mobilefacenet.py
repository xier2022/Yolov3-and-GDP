# -*- coding: utf-8 -*-



# 这段代码是一个用于目标检测的Python程序，使用了YOLO（You Only Look Once）算法。
# 首先，它导入了一些必要的库，如argparse、yolo和PIL。
# 然后定义了一个名为detect_img的函数，该函数从用户那里获取图像文件名，并使用YOLO模型对图像进行目标检测。
# 最后，根据命令行参数选择图像检测模式或视频检测模式。
# 在主函数中，首先定义了一个argparse.ArgumentParser对象，用于解析命令行参数。
# 然后添加了一些命令行选项，如模型权重文件路径、锚点定义文件路径、类定义文件路径、使用的GPU数量等。
# 接下来，根据FLAGS中的参数设置，选择图像检测模式或视频检测模式。
# 如果选择了图像检测模式，将调用detect_img函数；如果选择了视频检测模式，将调用detect_video函数。
import argparse
from yolo import YOLO, detect_video
from PIL import Image

def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors_path', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes_path', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,default="",
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    # elif "input" in FLAGS:
    #     detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
        # print("Must specify at least video_input_path.  See usage with --help.")
