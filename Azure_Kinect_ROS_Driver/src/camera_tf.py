#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import roslib
# roslib.load_manifest('learning_tf')
# import rospy

# import tf

# if __name__ == '__main__':
#     rospy.init_node('my_tf_broadcaster')
#     br = tf.TransformBroadcaster()
#     rate = rospy.Rate(10.0)
#     while not rospy.is_shutdown():
#         br.sendTransform((0.0, 2.0, 0.0),
#                          (0.0, 0.0, 0.0, 1.0),
#                          rospy.Time.now(),
#                          "camera_base",
#                          "base_link")
#         rate.sleep()
  
#import roslib
#roslib.load_manifest('learning_tf')

import rospy
import tf

if __name__ == '__main__':
    rospy.init_node('camera_tf_broadcaster')
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        br.sendTransform((-0.65, 0.0, 1.3),
                         tf.transformations.quaternion_from_euler(-1.57, 0, 1.57),
                         rospy.Time.now(),
                         "camera_base",
                         "base_link")
        rate.sleep()
