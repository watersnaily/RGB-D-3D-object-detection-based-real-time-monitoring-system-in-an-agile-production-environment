# RGB-D 3D object detection based real-time monitoring system in an agile production environment

## Azure Kinect ROS Driver
 This part uses the project [Azure Kinect ROS Driver](https://github.com/microsoft/Azure_Kinect_ROS_Driver), which is a
node that publishes sensor data from the [Azure Kinect Developer Kit](https://azure.microsoft.com/en-us/services/kinect-dk/)
to the [Robot Operating System (ROS)](https://www.ros.org/). This part modified several of the files according to actual needs. 
The modified files are as follows:
```
Azure_Kinect_ROS_Driver
├── calibration
│   ├── color_camera_calibration.txt
│   ├── depth_camera_calibration.txt
├── launch
│   ├── rectify.launch
├── src
│   ├── camera_tf.py
│   ├── k4a_calibration_transform_data.cpp
```
When using this part, first install the driver according to [Azure Kinect ROS Driver](https://github.com/microsoft/Azure_Kinect_ROS_Driver),
and then add or replace files in the corresponding file locations.  
After installing and modifying the driver, run `rectify.launch` to get the corrected images and other information.
If it is in an environment without a screen, before running `rectify.launch`, the DISPLAY variable need to be exported like this:  
`export DISPLAY:=0`

## object detector
A real-time 3D object detector. The main function is object_detector/src/detection_realtime.py.  
Run `detection_realtime.py` to detect objects in real time.

### Requirements
Test on
* Ubuntu-18.04
* ROS-melodic
* python 3.6
* CUDA-10.1
* Pytorch 1.6
* detectron2
* OpenCV 3.4.2

And cv_bridge cannot be used with python3 and ros environment, it can be solved by building cv_bridge with python3,
refer to [stackoverflow](https://stackoverflow.com/questions/49221565/unable-to-use-cv-bridge-with-ros-kinetic-and-python3).  

### Training data
```
object_detector
├── src
│   ├── train
│   │   ├── fasterR-CNN
│   │   │   ├── station_coco
│   │   │   ├── station_mask
│   │   ├── frustum_pointnets
│   │   │   ├── sunrgbd
│   │   ├── svm_classifier
│   │   │   ├── rgb
```
Put the data into the corresponding folders, and then run the corresponding program to train the model.

### Usage
First start the Azure Kinect ROS Driver node, which will publish sensor data,
and then run `object_detector/src/detection_realtime.py` for object detection.

## object detector cpu
This is the cpu version of object detector, which can be used on devices without gpu.
Its training data, models, and usage methods are consistent with the gpu version.
It is slightly different in terms of environmental requirements, Pytorch and detectron2 require cpu version, and it does not require CUDA.