# deeplabv1_ros 
Deeplab segmantic ROS wrapper, ROS中使用DeepLab模型。有关DeepLab的信息在这个[网站](https://github.com/tensorflow/models/tree/master/research/deeplab)可以看到.
![20220215144053](https://raw.githubusercontent.com/zhuhu00/img/master/20220215144053.png)
推理部分的代码主要参考的是这里：[Colab notebook](https://colab.sandbox.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb).

# 模型文件解释
`tar.gz`压缩包内包含有3个文件：pb文件是实际测试的模型，`ckpt`和`index`是预训练模型。主要看[models](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md)

# How to build
根据下面的步骤进行安装，需要注意以下部分需要安装`catkin_simple`的包才能够编译成功。

此外，代码中使用的是python2，tensorflow(1.15.2)进行推理的，这里需要安装对应python2的版本，可能出现的错误：

![20220105164942](https://raw.githubusercontent.com/zhuhu00/img/master/20220105164942.png)
解决办法是重新安装protobuf：
```bash
pip install protobuf==3.17.3
```

## Getting started
Clone this repository to the `src` folder of your catkin workspace, build your workspace and source it.

```bash
cd <catkin_ws>/src
git clone https://github.com/zhuhu00/deeplabv1_ros.git
catkin_make
source <catkin_ws>/devel/setup.bash
```

## Example usage
An example launch file is included processing a sequence from the [Freiburg RGB-D SLAM Dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset/download).

```bash
cd <catkin_ws>/src/deeplab_ros
chmod +x scripts/download_freiburg_rgbd_example_bag.sh 
./scripts/download_freiburg_rgbd_example_bag.sh
roslaunch deeplab_ros freiburg.launch
```

## ROS node

#### Parameters:

* **`~rgb_input`** [_string_]

    Topic name of the input RGB stream.

    Default: `"/camera/rgb/image_color"`

* **`~model`** [_string_]

    Name of the backbone network used for inference. List of available models: {"mobilenetv2_coco_voctrainaug", "mobilenetv2_coco_voctrainval", "xception_coco_voctrainaug", "xception_coco_voctrainval"}.
    If the specified model file doesn't exist, the node automatically downloads the file.

    Default: `"mobilenetv2_coco_voctrainaug"`
    
* **`~visualize`** [_bool_]

    If true, the segmentation result overlaid on top of the input RGB image is published to the `~segmentation_viz` topic.

    Default: `true`
    
        
#### Topics subscribed:

* topic name specified by parameter **`~rgb_input`** (default: **`/camera/rgb/image_color`**) [_sensor_mgs/Image_]

    Input RGB image to be processed.

    
#### Topics published:

* **`~segmentation`** [_sensor_mgs/Image_]

    Segmentation result.


* **`~segmentation_viz`** [_sensor_mgs/Image_]

    Visualization-friendly segmentation result color coded with the PASCAL VOC 2012 color map overlaid on top of the input RGB image.
