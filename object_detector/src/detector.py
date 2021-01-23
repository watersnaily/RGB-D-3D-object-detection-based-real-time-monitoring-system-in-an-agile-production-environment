import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'frustum_pointnets'))

import rospy
import torch
import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.catalog import Metadata

from depth2pc import depth2pc
import provider_fpointnet as provider

from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER
from model_util import get_box3d_corners
import torch.nn.functional as F
from frustum_pointnets import FrustumPointNetv1

from viz_util import point_cloud_publisher,bbox_publisher,box3d_corners,pose_publisher
from fine_pose import generate_station_model_with_normal,cicp
# from icp import cicp,icp

setup_logger()

class detector:
    def __init__(self,rgb_image,depth_image,fx,fy,cx,cy):
        self.set_up_faster_rcnn()
        self.set_up_fpointnet()
        self.detection(rgb_image,depth_image,fx,fy,cx,cy)
        
    def set_up_faster_rcnn(self):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = "weights/model_final.pth"
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.65
        self.predictor = DefaultPredictor(self.cfg)
        self.metadata = Metadata()
        self.metadata.set(thing_classes = ['station', 'forklift'])

    def set_up_fpointnet(self):
        self.FrustumPointNet = FrustumPointNetv1(n_classes=6,n_channel=6).cuda()
        self.pth = torch.load("weights/frustum_model.pth")
        self.FrustumPointNet.load_state_dict(self.pth['model_state_dict'])
        self.model = self.FrustumPointNet.eval()

    def detection(self,rgb_image,depth_image,fx,fy,cx,cy):
        print('start detection')
        rgb_image = rgb_image
        depth_image = np.nan_to_num(depth_image,nan=0)
        outputs = self.predictor(rgb_image)
        prob_list = outputs["instances"].scores
        class_list = outputs["instances"].pred_classes
        box2d_list = outputs["instances"].pred_boxes.tensor

        # # v = Visualizer(rgb_image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        # v = Visualizer(rgb_image[:, :, ::-1], metadata=self.metadata, scale=1.2)
        # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2.namedWindow('test',0)
        # cv2.imshow('test',out.get_image()[:, :, ::-1])
        # cv2.waitKey(0)

        # print("depth mean {}".format(np.mean(depth_image)))

        pitch = 0.09557043068606919
        rotation = np.array([[1,0,0],
                    [0,np.cos(pitch),-np.sin(pitch)],
                    [0,np.sin(pitch),np.cos(pitch)]])

        count = 0
        pose = np.zeros([1,4])
        pose_list = []

        for idx in range(len(class_list)):

            object_class = class_list[idx].cpu().numpy()
            prob = prob_list[idx].cpu().numpy()
            xmin,ymin,xmax,ymax = map(int,box2d_list[idx])

            if (xmax-xmin) > 1.5*(ymax-ymin):
                continue

            rgb = np.zeros_like(rgb_image)
            depth = np.zeros_like(depth_image)
            rgb[ymin:ymax,xmin:xmax] = rgb_image[ymin:ymax,xmin:xmax]
            depth[ymin:ymax,xmin:xmax] = depth_image[ymin:ymax,xmin:xmax]
            print("class: {} ,depth_mean: {}".format(object_class,np.mean(depth[ymin:ymax,xmin:xmax])))
            pcs = depth2pc(rgb, depth, fx, fy, cx, cy, 1).point_cloud_generator()
            pcs[:,0:3] = np.dot(pcs[:,0:3].astype(np.float32),rotation)
            mask = pcs[:,2]!=0
            pcs = pcs[mask,:]
            box2d_center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0])
            uvdepth = np.zeros((1,3))
            uvdepth[0,0:2] = box2d_center
            uvdepth[0,2] = np.mean(pcs[:,2])#20 # some random depth
            x = ((uvdepth[:,0]-cx)*uvdepth[:,2])/fx
            y = ((uvdepth[:,1]-cy)*uvdepth[:,2])/fy
            uvdepth[:,0] = x
            uvdepth[:,1] = y
            frustum_angle = -1 * np.arctan2(uvdepth[0,2], uvdepth[0,0]) # angle as to positive x-axis as in the Zoox paper
 
            # Pass objects that are too small
            if len(pcs)<5:
                continue

            if object_class == 0:
                object_class = 'box'
                data = provider.FrustumDataset(npoints=2048,pcs=pcs,object_class=object_class,frustum_angle=frustum_angle,prob=prob)
                point_set, rot_angle,prob, one_hot_vec = data.data()
                point_set = torch.unsqueeze(torch.tensor(point_set),0).transpose(2, 1).float().cuda()
                one_hot_vec = torch.unsqueeze(torch.tensor(one_hot_vec),0).float().cuda()

                # print('start fpointnets')
                logits, mask, stage1_center, center_boxnet, object_pts, \
                heading_scores, heading_residuals_normalized, heading_residuals, \
                size_scores, size_residuals_normalized, size_residuals, center = \
                self.model(point_set, one_hot_vec)

                corners_3d = get_box3d_corners(center,heading_residuals,size_residuals)

                logits = logits.cpu().detach().numpy()
                mask = mask.cpu().detach().numpy()
                center_boxnet = center_boxnet.cpu().detach().numpy()
                object_pts = object_pts.cpu().detach().squeeze().numpy().transpose(1,0)
                stage1_center = stage1_center.cpu().detach().numpy()
                center = center.cpu().detach().numpy()
                heading_scores = heading_scores.cpu().detach().numpy()
                # heading_residuals_normalized = heading_residuals_normalized.cpu().detach().numpy()
                heading_residuals = heading_residuals.cpu().detach().numpy()
                size_scores = size_scores.cpu().detach().numpy()
                size_residuals = size_residuals.cpu().detach().numpy()
                corners_3d = corners_3d.cpu().detach().numpy()

                output = np.argmax(logits, 2)
                heading_class = np.argmax(heading_scores)
                size_class = np.argmax(size_scores)
                corners_3d = corners_3d[0,heading_class,size_class]
                pred_angle = provider.class2angle(heading_class, heading_residuals[0, heading_class], NUM_HEADING_BIN)
                pred_size = provider.class2size(size_class,size_residuals[0,size_class])
                
                cloud = pcs[:,0:3].astype(np.float32)

                object_cloud = (object_pts-center_boxnet.repeat(object_pts.shape[0],0)).astype(np.float32)

                station_size = (0.979,0.969,0.979)
                cube = generate_station_model_with_normal(np.array([[0,0,0]]),station_size,-pred_angle)
                # object_cloud = rotate_pc_along_y(object_cloud,pred_angle)
                # cube = generate_station_model_with_normal(np.array([[0,0,0]]),station_size,0)
                station_cloud = cube.generate_points().astype(np.float32)
                # # object_cloud_crop = object_cloud[object_cloud[:,1]>(center[0][1]-0.48)]
                # # station_cloud_crop = station_cloud[station_cloud[:,1]>(center[0][1]-0.48)]
                # object_cloud_crop = object_cloud[object_cloud[:,1]>(-0.48)]
                # station_cloud_crop = station_cloud[station_cloud[:,1]>(-0.48)]
                # # station_cloud = rotate_pc_along_y(station_cloud, rot_angle)
                # cloud_icp = cicp(object_cloud,station_cloud_crop,max_iterations=30)
                cloud_icp = cicp(object_cloud,station_cloud,max_iterations=20)
                # cloud_icp = icp(object_cloud,station_cloud,max_iterations=20)
                T,R,t = cloud_icp.cicp()
                # R_angle = np.arccos(R[0,0])
                # print(R)
                # print(t)
                # if abs(t[0]) > abs(t[2]):
                #     t[2] = 0
                # if abs(t[0]) < abs(t[2]):
                #     t[0] = 0
                # print(t)
                cloud_t = np.tile(t,(station_cloud.shape[0],1))
                # station_cloud_rect = np.dot(station_cloud[:,:3]-cloud_t, np.transpose(np.linalg.pinv(R)))
                # station_cloud_rect = np.dot(station_cloud[:,:3]-cloud_t, np.transpose(np.linalg.pinv(R)))
                # station_cloud[:,:3] = rotate_pc_along_y(station_cloud[:,:3],-R_angle)
                station_cloud_rect = station_cloud[:,:3]-cloud_t

                station_cloud_rect = station_cloud_rect + center.repeat(station_cloud_rect.shape[0],0)
                object_cloud = object_cloud + center.repeat(object_cloud.shape[0],0)
                station_cloud[:,:3] = station_cloud[:,:3] + center.repeat(station_cloud.shape[0],0)
                
                center = center - t[np.newaxis, :]
                corners_3d_rect = box3d_corners(center,pred_angle,station_size)


                object_cloud = rotate_pc_along_y(object_cloud,-rot_angle)
                station_cloud_rect = rotate_pc_along_y(station_cloud_rect,-rot_angle)
                station_cloud[:,:3] = rotate_pc_along_y(station_cloud[:,:3],-rot_angle)
                center = rotate_pc_along_y(center,-rot_angle)
                corners_3d = rotate_pc_along_y(corners_3d,-rot_angle)
                corners_3d_rect = rotate_pc_along_y(corners_3d_rect,-rot_angle)

                center[0,1] = 0.815

                pose[0,:3] = center
                pose[0,3] = pred_angle + rot_angle
                pose_list.append(pose.copy())
                
                
                
                # # station_cloud_rect = rotate_pc_along_y(station_cloud[:,:3]-cloud_t, R_angle)
                # # cloud = np.dot(cloud, np.transpose(R))+np.tile(t,(cloud.shape[0],1))
                # object_pub = point_cloud_publisher('/points_object',object_cloud_crop)
                # station_pub = point_cloud_publisher('/points_station',station_cloud_crop[:,:3])
                count += 1
                # station_pub = point_cloud_publisher('/points_station%d'%(count),station_cloud[:,:3])
                station_rect_pub = point_cloud_publisher('/points_station_rect%d'%(count),station_cloud_rect)
                # bbox_pub = bbox_publisher('/bbox%d'%(count),corners_3d)
                bbox_pub_rect = bbox_publisher('/bbox_rect%d'%(count),corners_3d_rect,color = "green")
                object_pub = point_cloud_publisher('/points_object%d'%(count),object_cloud)
                # cloud_pub = point_cloud_publisher('/points_cloud%d'%(count),cloud)
                # cloud_pub1 = point_cloud_publisher('/points_cloud1',cloud1)

                station_rect_pub.point_cloud_publish()
                bbox_pub_rect.bbox_publish()
                object_pub.point_cloud_publish()

                # bbox_list.append(bbox_pub)
                # bbox_list.append(bbox_pub_rect)
                # point_cloud_list.append(station_pub)
                # point_cloud_list.append(station_rect_pub)
                # point_cloud_list.append(object_pub)
                # point_cloud_list.append(cloud_pub)
                # with open('results.txt','ab') as f:
                #     np.savetxt(f, box_info, delimiter=" ") 

                # rate = rospy.Rate(10)
                # while not rospy.is_shutdown():
                #     # point_cloud_list[3].point_cloud_publish()
                # #             # object_pub.point_cloud_publish()
                #     station_pub.point_cloud_publish()
                #     station_rect_pub.point_cloud_publish()
                #     bbox_pub.bbox_publish()
                #     bbox_pub_rect.bbox_publish()
                #     object_pub.point_cloud_publish()
                #     cloud_pub.point_cloud_publish()
                # #             # cloud_pub1.point_cloud_publish()
                #     rate.sleep()

        pose_pub = pose_publisher('station_pose',pose_list)
        pose_pub.pose_publish()
        print('once detection')
        # print(len(bbox_list))
        # print(len(point_cloud_list))
        # rate = rospy.Rate(10)
        # while not rospy.is_shutdown():
        #     for bbox in bbox_list:
        #         bbox.bbox_publish()
        #     for point_cloud in point_cloud_list:
        #         point_cloud.point_cloud_publish()
        #     rate.sleep()

def rotate_pc_along_y(pc,rot_angle):
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc