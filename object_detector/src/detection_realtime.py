#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2

import rospy
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from sensor_msgs.msg import Image,CameraInfo

from detector import detector

class realtime_detector:
    def __init__(self):    
        
        self.bridge = CvBridge()
        self.rgb_sub = message_filters.Subscriber("/k4a/rgb/image_rect_color", Image, queue_size=1)
        self.depth_sub = message_filters.Subscriber("/k4a/depth_to_rgb/image_rect", Image, queue_size=1)
        self.cam_sub = message_filters.Subscriber("/k4a/rgb/camera_info", CameraInfo, queue_size=1)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub, self.cam_sub], 10, 0.1, allow_headerless = True)
        self.ts.registerCallback(self.callback)

    def callback(self,rgb_img,depth_img,camera_info):
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_img, "bgra8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_img, "32FC1")
        except CvBridgeError as e:
            print(e)

        rgb_image = rgb_image[:,:,:-1]
        k = camera_info.K
        fx = k[0]
        fy = k[4]
        cx = k[2]
        cy = k[5]

        detector(rgb_image,depth_image,fx,fy,cx,cy)

if __name__ == '__main__':
    try:
        # 初始化ros节点
        rospy.init_node("detection_realtime")
        rospy.loginfo("Starting realtime detection node")
        realtime_detector()
        rospy.spin()

    except KeyboardInterrupt:
        print("Shutting down realtime detection node.")