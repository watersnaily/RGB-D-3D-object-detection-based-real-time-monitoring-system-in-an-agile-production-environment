import rospy
import numpy as np

from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import std_msgs.msg

from visualization_msgs.msg import Marker,MarkerArray
from geometry_msgs.msg import Point,PoseArray,Pose



class point_cloud_publisher:

    def __init__(self,pub_topic,cloud_arr):  
        self.pcs_pub = rospy.Publisher(pub_topic, PointCloud2, queue_size=1000000)
        self.cloud_msg = self.array_to_pointcloud2(cloud_arr)

        # rate = rospy.Rate(10)
        # while not rospy.is_shutdown():
        #     self.pcs_pub.publish(cloud_msg)
        #     rate.sleep()

    def point_cloud_publish(self):
        self.pcs_pub.publish(self.cloud_msg)

    def array_to_pointcloud2(self,cloud_arr):
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "camera_base"
        cloud_msg = point_cloud2.create_cloud_xyz32(header, cloud_arr)
        return cloud_msg



class bbox_publisher:

    def __init__(self,pub_topic,corners,color = 'red'):  
        self.marker_array_pub = rospy.Publisher(pub_topic, MarkerArray, queue_size=10)
        self.marker_array = MarkerArray()
        self.generate_marker_array(corners,color)

    def bbox_publish(self):
        self.marker_array_pub.publish(self.marker_array)

    def generate_marker_array(self,corners,color):
 
        idx = 0
        for i in range(4):
            marker = self.rosMarker(corners[i%4],corners[(i+1)%4], idx, color)
            idx += 1
            self.marker_array.markers.append(marker)
            marker = self.rosMarker(corners[i%4+4],corners[(i+1)%4+4], idx, color)
            idx += 1
            self.marker_array.markers.append(marker)
            marker = self.rosMarker(corners[i],corners[i+4], idx, color)
            idx += 1
            self.marker_array.markers.append(marker)

        # rate = rospy.Rate(10) 

        # while not rospy.is_shutdown():
        #     self.marker_array_pub.publish(marker_array)
        #     rate.sleep()

    def rosMarker(self,p1,p2,idx,color):
        val = 0.1
        if (color == "green"):
            val = 0.9

        marker = Marker()
        # marker.header.frame_id = "camera_base"
        marker.header.frame_id = "camera_base"
        marker.type = marker.LINE_LIST
        marker.action = marker.ADD
        marker.scale.x = 0.01
        # marker.color.a = 0.4
        marker.color.a = 1
        # marker.color.r = 1
        marker.color.r = 1 - val
        marker.color.g = val
        marker.color.b = 0.2

        point1 = Point()
        point2 = Point()
        point1.x = p1[0]
        point1.y = p1[1]
        point1.z = p1[2]
        point2.x = p2[0]
        point2.y = p2[1]
        point2.z = p2[2]
        # print("type is {}".format(type(point1)))
        # print(point1)

        marker.points.append(point1)
        marker.points.append(point2)
        # marker.lifetime = rospy.Duration(0.3)
        marker.id = idx
        return marker

class pose_publisher:

    def __init__(self,pub_topic,pose_list):
        self.pose_array_pub = rospy.Publisher(pub_topic, PoseArray, queue_size=10)
        self.pose_array = PoseArray()
        self.pose_array.header.frame_id = "base_link"
        self.pose_array.header.stamp = rospy.Time.now()
        self.generate_pose_array(pose_list)

    def pose_publish(self):
        self.pose_array_pub.publish(self.pose_array)

    def generate_pose_array(self,pose_list):
 
        for p in pose_list:
            pose = Pose()
            x,y,z,w = self.rpy2quaternion(0,0,-p[0,3])
            pose.position.x = -p[0,2] - 0.65
            pose.position.y = p[0,0]
            pose.position.z = -p[0,1] + 1.3
            pose.orientation.x = x
            pose.orientation.y = y
            pose.orientation.z = z
            pose.orientation.w = w

            self.pose_array.poses.append(pose)

    def rpy2quaternion(self,roll, pitch, yaw):
        x=np.sin(pitch/2)*np.sin(yaw/2)*np.cos(roll/2)+np.cos(pitch/2)*np.cos(yaw/2)*np.sin(roll/2)
        y=np.sin(pitch/2)*np.cos(yaw/2)*np.cos(roll/2)+np.cos(pitch/2)*np.sin(yaw/2)*np.sin(roll/2)
        z=np.cos(pitch/2)*np.sin(yaw/2)*np.cos(roll/2)-np.sin(pitch/2)*np.cos(yaw/2)*np.sin(roll/2)
        w=np.cos(pitch/2)*np.cos(yaw/2)*np.cos(roll/2)-np.sin(pitch/2)*np.sin(yaw/2)*np.sin(roll/2)
        return x, y, z, w

def box3d_corners(centers, headings, sizes):
    """ Input: (1,3), , (3), Output: (8,3) """
    l = sizes[0]
    w = sizes[1]
    h = sizes[2]
    x_corners = np.array([l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2])# (8)
    y_corners = np.array([h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]) # (8)
    z_corners = np.array([w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]) # (8)
    corners = np.concatenate([x_corners.reshape(1,8), y_corners.reshape(1,8),\
                            z_corners.reshape(1,8)], axis=0) # (3,8)

    ###ipdb.set_trace()
    #print x_corners, y_corners, z_corners
    c = np.cos(headings)
    s = np.sin(headings)
    R = np.array([[c,0,s],[0,1,0],[-s,0,c]])
    corners_3d = np.dot(R,corners).transpose() 
    corners_3d = corners_3d + centers.repeat(corners_3d.shape[0],0)
    return corners_3d 