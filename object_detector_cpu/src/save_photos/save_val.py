#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image,CameraInfo,PointCloud2
from sensor_msgs import point_cloud2
import message_filters
# import ctypes
# import struct


import pcl


class save_photos:
    def __init__(self):    

        self.bridge = CvBridge()
        self.count = 0
        self.rgb_sub = message_filters.Subscriber("/k4a/rgb/image_rect_color", Image, queue_size=1)
        self.depth_sub = message_filters.Subscriber("/k4a/depth_to_rgb/image_rect", Image, queue_size=1)
        # self.cam_sub = message_filters.Subscriber("/k4a/rgb/camera_info", CameraInfo, queue_size=1)
        self.pcs_sub = message_filters.Subscriber("/k4a/points2", PointCloud2, queue_size=1)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub, self.pcs_sub], 10, 0.1, allow_headerless = True)
        self.ts.registerCallback(self.callback)
        rospy.spin()
    
    def callback(self,rgb, depth,pointclouds):

        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb, "bgra8")
            pcs = point_cloud2.read_points(pointclouds, skip_nans=True)
            depth_image = self.bridge.imgmsg_to_cv2(depth, "32FC1")
            # frame = np.array(cv_image, dtype=np.uint8)
        except CvBridgeError as e:
            print (e)
 
        cv2.namedWindow('test',0)
        cv2.imshow('test',rgb_image)
        # cv2.namedWindow("Image window",0)
        # cv2.resizeWindow("Image window",rgb_image.shape[1]/2,rgb_image.shape[0]/2)
        # cv2.imshow("Image window", rgb_image)
        # cv2.imshow("Image window", depth_image)

        self.k = cv2.waitKey(3)

        photos_dir = '../../photos/val'

        if self.k & 0xFF == ord('s'):
            self.count += 1
            points_list = []
            for data in pcs:
                points_list.append([data[0], data[1], data[2], data[3]])

            # print(len(points_list))
            print(self.count)
            pcl_data = pcl.PointCloud_PointXYZRGB()
            pcl_data.from_list(points_list)
            pcl.save(pcl_data,os.path.join(photos_dir,'point_clouds','%06d.pcd'%(self.count)))
            cv2.imwrite(os.path.join(photos_dir,'rgb','%06d.png'%(self.count)),rgb_image)
            np.savetxt(os.path.join(photos_dir,'depth','%06d.txt'%(self.count)), depth_image,fmt='%f',delimiter=' ')



if __name__ == '__main__':
    try:
        # 初始化ros节点
        rospy.init_node("save_val_photos")
        rospy.loginfo("Starting save val photos node")
        save_photos()
        #rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down save val photos node.")
        cv2.destroyAllWindows()
