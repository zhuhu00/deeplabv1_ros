#!/usr/bin/env python3
import os
import sys
import tempfile
import threading
from six.moves import urllib

import PIL
import numpy as np
import cv2

import tensorflow as tf

from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image

# from deeplabv1_ros import deeplab
import deeplab

print(sys.version,"======")

class DeepLabNode(object):
    def __init__(self):
        self._cv_bridge = CvBridge()
        self._last_msg = None
        self._msg_lock = threading.Lock()

        self._publish_rate = rospy.get_param('~publish_rate', 100)
        self._visualize = rospy.get_param('~visualize', True)

        rgb_input = rospy.get_param('~rgb_input', '/camera/rgb/image_color')

        rospy.Subscriber(rgb_input, Image, self._image_callback, queue_size=1)

        self.label_pub = rospy.Publisher('~segmentation', Image, queue_size=1)
        self.vis_pub = rospy.Publisher('~segmentation_viz', Image, queue_size=1)

        MODEL_NAME = rospy.get_param('~model', 'mobilenetv2_coco_voctrainaug')

        _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
        _MODEL_URLS = {
            'mobilenetv2_coco_voctrainaug':
                'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
            'mobilenetv2_coco_voctrainval':
                'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
            'xception_coco_voctrainaug':
                'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
            'xception_coco_voctrainval':
                'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
        }

        # _TARBALL_NAME = 'deeplab_model.tar.gz'
        _TARBALL_NAME = _MODEL_URLS[MODEL_NAME]

        # download the weights file in temp folder
        # model_dir = tempfile.mkdtemp()
        model_dir = rospy.get_param('models_files','../weights')
        tf.gfile.MakeDirs(model_dir)

        download_path = os.path.join(model_dir, _TARBALL_NAME)
        print(download_path)

        if(not os.path.exists(download_path)):
            print('Downloading model, this might take a while...')
            urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
                           download_path)
            print('Download completed! Loading DeepLab model...')

        print("==Using the model: "+MODEL_NAME )
        self._model = deeplab.DeepLabModel(download_path)
        print('Model loaded successfully!')

    def run(self):
        rate = rospy.Rate(self._publish_rate)

        while not rospy.is_shutdown():
            if self._msg_lock.acquire(False):
                msg = self._last_msg
                self._last_msg = None
                self._msg_lock.release()
            else:
                rate.sleep()
                continue

            if msg is not None:
                # rgb_image = self._cv_bridge.imgmsg_to_cv2(msg, "passthrough") # 这里有错误,会导向python2
                rgb_image = self.imgmsg_to_cv2(msg)


                # Run detection.
                seg_map = self.detect(rgb_image)

                rospy.logdebug("Publishing semantic labels.")
                # rospy.loginfo("Publishing semantic labels.")
                # label_msg = self._cv_bridge.cv2_to_imgmsg(seg_map, 'mono16')
                label_msg = self.cv2_to_imgmsg(seg_map)
                label_msg.header = msg.header
                self.label_pub.publish(label_msg)

                if self._visualize:
                    # Overlay segmentation on RGB image.
                    image = self.visualize(rgb_image, seg_map)
                    label_color_msg = self.cv2_to_imgmsg(image)
                    label_color_msg.header = msg.header
                    self.vis_pub.publish(label_color_msg)


            rate.sleep()

    def detect(self, rgb_image):
        rgb_image = PIL.Image.fromarray(rgb_image)

        resized_im, seg_map = self._model.run(rgb_image)
        seg_map = cv2.resize(seg_map.astype(np.float32), rgb_image.size, interpolation = cv2.INTER_NEAREST).astype(np.uint16)

        return seg_map

    def visualize(self, rgb_image, seg_map, alpha = 0.6):
        image = rgb_image.copy()
        seg_image = deeplab.label_to_color_image(seg_map.astype(
            np.int64)).astype(np.uint8)
        cv2.addWeighted(seg_image, alpha, image, 1 - alpha, 0, image)
        return image


    def _image_callback(self, msg):
        rospy.logdebug("Got an image.")

        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._msg_lock.release()


    def cv2_to_imgmsg(self, cv_image):
        img_msg = Image()
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = "bgr8"
        img_msg.is_bigendian = 0
        img_msg.data = cv_image.tobytes()
        img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
        return img_msg
    

    def imgmsg_to_cv2(self, img_msg):
        if img_msg.encoding != "rgb8":
            rospy.logwarn_once("This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
        dtype = np.dtype("uint8") # Hardcode to 8 bits...
        dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
        image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                        dtype=dtype, buffer=img_msg.data)
        # If the byt order is different between the message and the system.
        if img_msg.is_bigendian == (sys.byteorder == 'little'):
            image_opencv = image_opencv.byteswap().newbyteorder()
        image_reverse = np.flip(image_opencv, axis=2)

        return image_reverse

def main():
    rospy.init_node('deeplabv1_ros_node')

    node = DeepLabNode()
    node.run()


if __name__ == '__main__':
    main()
