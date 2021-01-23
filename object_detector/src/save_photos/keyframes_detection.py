#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class keyframes_detector:
    def __init__(self):    

        self.bridge = CvBridge()
        self.threshold = 25
        self.sum_threshold = 2550
        self.count = 0
        self.prevFrame = None
        self.image_sub = rospy.Subscriber("/k4a/rgb/image_rect_color", Image, self.callback,queue_size=1)

    def callback(self,data):

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgra8")
            # frame = np.array(cv_image, dtype=np.uint8)
        except CvBridgeError as e:
            print e

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prevFrame is None:
            self.prevFrame = gray
            return  

        frameDelta = cv2.absdiff(self.prevFrame, gray)
        self.prevFrame = gray
        diff = cv2.threshold(frameDelta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        diff_sum = np.sum(diff)

        if diff_sum > self.sum_threshold:
            #cv2.circle(cv_image, (60, 60), 30, (0,0,255), -1)
            self.count += 1
            cv2.imwrite('../../photos/capture_color'+str(self.count)+'.png',cv_image)

        cv2.imshow("Image window", cv_image)

        cv2.waitKey(3)
  
if __name__ == '__main__':
    try:
        # 初始化ros节点
        rospy.init_node("keyframes_detection")
        rospy.loginfo("Starting keyframes detection node")
        keyframes_detector()
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down keyframes detection node."
        cv2.destroyAllWindows()
