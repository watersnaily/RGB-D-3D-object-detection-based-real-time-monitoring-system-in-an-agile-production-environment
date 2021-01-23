#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import thread
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class camera_caliration:
    def __init__(self, width, height, size):    
        
        self.bridge = CvBridge()
        self.width = width
        self.height = height
        self.size = size
        self.obj_points = []
        self.img_points = []
        self.shape = []
        self.count = 0
        self.k = 0
        self.start_calibration = False
        self.image_sub = rospy.Subscriber("/ir/image_raw", Image, self.callback)
        thread.start_new_thread(self.calibration, ())


    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "mono8")
        except CvBridgeError as e:
            print e

        if not self.start_calibration:
            cv_image = self.find_corners(cv_image)
        if self.shape == []:
            self.shape = (cv_image.shape[1],cv_image.shape[0])
        cv2.imshow('FoundCorners',cv_image)
        self.k = cv2.waitKey(1)
        if self.k & 0xFF == ord('q'):
            self.start_calibration = True

    def calibration(self):
        while True:	
            if self.start_calibration:
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_points, self.img_points, self.shape, None, None, flags=cv2.CALIB_RATIONAL_MODEL)
                print 'end calibration'
                print 'ret:'
                print (ret)
                print 'camera matrix:'
                print (mtx)
                print 'distortion coefficients:'
                print (dist)
                parameter = [mtx[0][0],mtx[0][2],mtx[1][1],mtx[1,2]] + dist[0][:8].tolist()
                print 'parameter:'
                print (parameter)
                np.savetxt("../../../Azure_Kinect_ROS_Driver/calibration/depth_camera_calibration.txt", parameter,fmt='%f',delimiter=' ')
                print 'save parameter'

                mean_error = 0
                for i in xrange(len(self.obj_points)):
                    img_points2, _ = cv2.projectPoints(self.obj_points[i], rvecs[i], tvecs[i], mtx, dist)
                    error = cv2.norm(self.img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
                    mean_error += error

                print "total error: ", mean_error / len(self.obj_points)
                
                break
        
        # rospy.loginfo("obj ponits size[%d], img points size[%d]",len(obj_points),len(img_points))

    def find_corners(self,image):
        w = self.width
        h = self.height
        s = self.size
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # cp_int: corner point in int form, save the coordinate of corner points in world sapce in 'int' form
        # like (0,0,0), (1,0,0), (2,0,0) ....,(10,7,0).
        cp_int = np.zeros((w*h,3), np.float32)
        cp_int[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
        # cp_world: corner point in world space, save the coordinate of corner points in world space.
        cp_world = cp_int*s
            
        # find the corners, cp_img: corner points in pixel space.
        ret, cp_img = cv2.findChessboardCorners(image, (w,h), None)
        # if ret is True, save.
        if ret == True:
            cv2.drawChessboardCorners(image, (w,h), cp_img, ret)
            if self.k & 0xFF == ord('s'):
                cv2.cornerSubPix(image,cp_img,(11,11),(-1,-1),criteria)
                self.obj_points.append(cp_world)
                self.img_points.append(cp_img)
                self.count += 1
                rospy.loginfo("the number of saved image: %d", self.count)

        return image

if __name__ == '__main__':
    try:
        # 初始化ros节点
        rospy.init_node("depth_camera_callibration")
        rospy.loginfo("Starting depth camera callibration node")

        camera_caliration(15,10,0.072)
        rospy.spin()

    except KeyboardInterrupt:
        print "Shutting down depth camera callibration node."
        cv2.destroyAllWindows()