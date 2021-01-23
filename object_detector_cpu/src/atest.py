import cv2
from detector import detector
import numpy as np
import rospy


# rgb_path = '/home/he/Kit/Masterarbeit/3D_Mapping_ws/src/object_detector/photos/val/rgb/000048.png'
# depth_path = '/home/he/Kit/Masterarbeit/3D_Mapping_ws/src/object_detector/photos/val/depth/000048.txt'
rgb_path =   '/home/he/Kit/Masterarbeit/3D_Mapping_ws/src/object_detector/photos/val/rgb/000001.png'
depth_path='/home/he/Kit/Masterarbeit/3D_Mapping_ws/src/object_detector/photos/val/depth/000001.txt'
fx = 979
fy = 979
cx = 1027
cy = 770

rgb_image = cv2.imread(rgb_path)
depth_image =  np.loadtxt(depth_path)


rospy.init_node("atest")
rate = rospy.Rate(10)
while not rospy.is_shutdown():
    detector(rgb_image,depth_image,fx,fy,cx,cy)
    rate.sleep()