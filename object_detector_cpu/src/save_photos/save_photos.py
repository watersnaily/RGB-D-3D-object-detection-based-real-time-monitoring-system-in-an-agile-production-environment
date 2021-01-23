#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class save_photos:
    def __init__(self,photo_topic,save_pth):    

        self.bridge = CvBridge()
        self.count = 0
        self.k = 0
        self.save_pth = save_pth
        self.image_sub = rospy.Subscriber(photo_topic, Image, self.callback,queue_size=1)

    def callback(self,data):

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgra8")
	    #cv_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
            # frame = np.array(cv_image, dtype=np.uint8)
        except CvBridgeError as e:
            print (e)

        cv2.namedWindow("Image window",0)
        #cv2.resizeWindow("Image window",cv_image.shape[1]/2,cv_image.shape[0]/2)
        cv2.imshow("Image window", cv_image)

        self.k = cv2.waitKey(3)

        if self.k & 0xFF == ord('s'):
            self.count += 1
            # cv2.imwrite('/photos/calibration_color/pose/photo_'+str(self.count)+'.png',cv_image)
            cv2.imwrite(os.path.join(self.save_pth,'{:0>6d}.png'.format(self.count)),cv_image)
  
if __name__ == '__main__':
    try:
        # 初始化ros节点
        rospy.init_node("save_photos")
        rospy.loginfo("Starting save photos node")
        photo_topic = "/k4a/rgb/image_rect_color"
        save_pth = "../../pohtos/save"
        save_photos(photo_topic,save_pth)
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting save photos node.")
        cv2.destroyAllWindows()
