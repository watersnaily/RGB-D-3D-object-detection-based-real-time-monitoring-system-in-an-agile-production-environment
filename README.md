# RGB-D 3D object detection based real-time monitoring system in an agile production environment

## Azure Kinect ROS Driver
 This part uses the project [Azure Kinect ROS Driver](https://github.com/microsoft/Azure_Kinect_ROS_Driver), which is a
node that publishes sensor data from the [Azure Kinect Developer Kit](https://azure.microsoft.com/en-us/services/kinect-dk/)
to the [Robot Operating System (ROS)](https://www.ros.org/). This part modified several of the files according to actual needs. 
The modified files are as follows:
'''
Azure_Kinect_ROS_Driver
├── calibration
│   ├── color_camera_calibration.txt
│   ├── depth_camera_calibration.txt
├── launch
│   ├── rectify.launch
├── src
│   ├── camera_tf.py
│   ├── k4a_calibration_transform_data.cpp
'''
When using this part, first install the driver according to [Azure Kinect ROS Driver](https://github.com/microsoft/Azure_Kinect_ROS_Driver),
and then add or replace files in the corresponding file locations.
