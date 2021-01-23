''' Provider class and helper functions for Frustum PointNets.
Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

import sys
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# from box_util import box3d_iou

# import ipdb

NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 6 # one cluster for each type

g_type2class={'table':0,'chair':1,'desk':2,'shelf':3,'cabinet':4,'box':5}
g_class2type = {g_type2class[t]:t for t in g_type2class}
g_type2onehotclass = {'table':0,'chair':1,'desk':2,'shelf':3,'cabinet':4,'box':5}

g_type_mean_size = {'table': np.array([0.89078179,1.4377661,0.71774923]),
                    'chair': np.array([0.58838503,0.55581025,0.84174869]),
                    'desk': np.array([0.72018677,1.4049219,0.79348079]),
                    'shelf':np.array([0.43792965,1.41958785,0.94123454]),
                    'cabinet':np.array([0.57207646,1.37302015,1.21174823]),
                    'box':np.array([0.37163918,0.39581311,0.3400343 ])
                    }

try:
    raw_input  # Python 2
except NameError:
    raw_input = input  # Python 3


def rotate_pc_along_y(pc, rot_angle):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc




def class2angle(pred_cls, residual, num_class, to_label_format=True):
    ''' Inverse function to angle2class.
    If to_label_format, adjust angle to the range as in labels.
    '''
    angle_per_class = 2 * np.pi / float(num_class)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle > np.pi:
        angle = angle - 2 * np.pi
    return angle


def size2class(size, type_name):
    ''' Convert 3D bounding box size to template class and residuals.
    todo (rqi): support multiple size clusters per type.

    Input:
        size: numpy array of shape (3,) for (l,w,h)
        type_name: string
    Output:
        size_class: int scalar
        size_residual: numpy array of shape (3,)
    '''
    size_class = g_type2class[type_name]
    size_residual = size - g_type_mean_size[type_name]
    return size_class, size_residual


def class2size(pred_cls, residual):
    ''' Inverse function to size2class. '''
    mean_size = g_type_mean_size[g_class2type[pred_cls]]
    return mean_size + residual


class FrustumDataset(object):
    ''' Dataset class for Frustum PointNets training/evaluation.
    Load prepared KITTI data from pickled files, return individual data element
    [optional] along with its annotations.
    '''

    def __init__(self, npoints, pcs, object_class, frustum_angle, prob,random_flip=False, 
                        random_shift=False, rotate_to_center=True, one_hot=True):
        '''
        Input:
            npoints: int scalar, number of points for frustum point cloud.
            random_flip: bool, in 50% randomly flip the point cloud
                in left and right (after the frustum rotation if any)
            random_shift: bool, if True randomly shift the point cloud
                back and forth by a random distance
            rotate_to_center: bool, whether to do frustum rotation
            one_hot: bool, if True, return one hot vector
        '''
        self.npoints = npoints
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.rotate_to_center = rotate_to_center
        self.one_hot = one_hot
        self.pcs = pcs
        self.object_class = object_class
        # frustum_angle is clockwise angle from positive x-axis
        self.frustum_angle = frustum_angle
        self.prob = prob

    def data(self):
        # ------------------------------ INPUTS ----------------------------
        rot_angle = self.get_center_view_rot_angle()
        # np.pi/2.0 + self.frustum_angle_list [index]float,[-pi/2,pi/2]

        # Compute one hot vector
        if self.one_hot:  # True
            cls_type = self.object_class
            # assert (cls_type in ['Car', 'Pedestrian', 'Cyclist'])
            one_hot_vec = np.zeros((6))
            one_hot_vec[g_type2onehotclass[cls_type]] = 1

        # Get point cloud
        if self.rotate_to_center:  # True
            point_set = self.get_center_view_point_set()  #pts after Frustum rotation
        else:
            point_set = self.pcs

        # ipdb.set_trace()
        # Resample
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice, :]

        if self.one_hot:
            return point_set, rot_angle, self.prob, one_hot_vec
        else:
            return point_set, rot_angle, self.prob

    def get_center_view_rot_angle(self):
        ''' Get the frustum rotation angle, it isshifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        return np.pi / 2.0 + self.frustum_angle

    # def get_box3d_center(self, index):
    #     ''' Get the center (XYZ) of 3D bounding box. '''
    #     box3d_center = (self.box3d_list[index][0, :] + \
    #                     self.box3d_list[index][6, :]) / 2.0
    #     return box3d_center

    # def get_center_view_box3d_center(self, index):
    #     ''' Frustum rotation of 3D bounding box center. '''
    #     box3d_center = (self.box3d_list[index][0, :] + \
    #                     self.box3d_list[index][6, :]) / 2.0
    #     return rotate_pc_along_y(np.expand_dims(box3d_center, 0), \
    #                              self.get_center_view_rot_angle()).squeeze()

    # def get_center_view_box3d(self, index):
    #     ''' Frustum rotation of 3D bounding box corners. '''
    #     box3d = self.box3d_list[index]
    #     box3d_center_view = np.copy(box3d)
    #     return rotate_pc_along_y(box3d_center_view, \
    #                              self.get_center_view_rot_angle())

    def get_center_view_point_set(self):
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.pcs)
        return rotate_pc_along_y(point_set, self.get_center_view_rot_angle())


# ----------------------------------
# Helper functions for evaluation
# ----------------------------------

# def get_3d_box(box_size, heading_angle, center):
#     ''' Calculate 3D bounding box corners from its parameterization.
#     Input:
#         box_size: tuple of (l,w,h)
#         heading_angle: rad scalar, clockwise from pos x axis
#         center: tuple of (x,y,z)
#     Output:
#         corners_3d: numpy array of shape (8,3) for 3D box cornders
#     '''

#     def roty(t):
#         c = np.cos(t)
#         s = np.sin(t)
#         return np.array([[c, 0, s],
#                          [0, 1, 0],
#                          [-s, 0, c]])

#     R = roty(heading_angle)
#     l, w, h = box_size
#     x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2];
#     y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2];
#     z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];
#     corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
#     corners_3d[0, :] = corners_3d[0, :] + center[0];
#     corners_3d[1, :] = corners_3d[1, :] + center[1];
#     corners_3d[2, :] = corners_3d[2, :] + center[2];
#     corners_3d = np.transpose(corners_3d)
#     return corners_3d


# def compute_box3d_iou(center_pred,
#                       heading_logits, heading_residuals,
#                       size_logits, size_residuals,
#                       center_label,
#                       heading_class_label, heading_residual_label,
#                       size_class_label, size_residual_label):
#     ''' Compute 3D bounding box IoU from network output and labels.
#     All inputs are numpy arrays.
#     Inputs:
#         center_pred: (B,3)
#         heading_logits: (B,NUM_HEADING_BIN)
#         heading_residuals: (B,NUM_HEADING_BIN)
#         size_logits: (B,NUM_SIZE_CLUSTER)
#         size_residuals: (B,NUM_SIZE_CLUSTER,3)
#         center_label: (B,3)
#         heading_class_label: (B,)
#         heading_residual_label: (B,)
#         size_class_label: (B,)
#         size_residual_label: (B,3)
#     Output:
#         iou2ds: (B,) birdeye view oriented 2d box ious
#         iou3ds: (B,) 3d box ious
#     '''
#     batch_size = heading_logits.shape[0]
#     heading_class = np.argmax(heading_logits, 1)  # B
#     heading_residual = np.array([heading_residuals[i, heading_class[i]] \
#                                  for i in range(batch_size)])  # B,
#     size_class = np.argmax(size_logits, 1)  # B
#     size_residual = np.vstack([size_residuals[i, size_class[i], :] \
#                                for i in range(batch_size)])

#     iou2d_list = []
#     iou3d_list = []
#     for i in range(batch_size):
#         heading_angle = class2angle(heading_class[i],
#                                     heading_residual[i], NUM_HEADING_BIN)
#         box_size = class2size(size_class[i], size_residual[i])
#         corners_3d = get_3d_box(box_size, heading_angle, center_pred[i])

#         heading_angle_label = class2angle(heading_class_label[i],
#                                           heading_residual_label[i], NUM_HEADING_BIN)
#         box_size_label = class2size(size_class_label[i], size_residual_label[i])
#         corners_3d_label = get_3d_box(box_size_label,
#                                       heading_angle_label, center_label[i])

#         iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_label)
#         iou3d_list.append(iou_3d)
#         iou2d_list.append(iou_2d)
#     return np.array(iou2d_list, dtype=np.float32), \
#            np.array(iou3d_list, dtype=np.float32)


# def from_prediction_to_label_format(center, angle_class, angle_res, \
#                                     size_class, size_res, rot_angle):
#     ''' Convert predicted box parameters to label format. '''
#     l, w, h = class2size(size_class, size_res)
#     ry = class2angle(angle_class, angle_res, NUM_HEADING_BIN) + rot_angle
#     tx, ty, tz = rotate_pc_along_y(np.expand_dims(center, 0), -rot_angle).squeeze()
#     ty += h / 2.0
#     return h, w, l, tx, ty, tz, ry


if __name__ == '__main__':
    import mayavi.mlab as mlab

    sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
    from viz_util import draw_lidar, draw_gt_boxes3d

    median_list = []
    dataset = FrustumDataset(1024, split='val',
                             rotate_to_center=True, random_flip=True, random_shift=True)
    for i in range(len(dataset)):
        data = dataset[i]
        print(('Center: ', data[2], \
               'angle_class: ', data[3], 'angle_res:', data[4], \
               'size_class: ', data[5], 'size_residual:', data[6], \
               'real_size:', g_type_mean_size[g_class2type[data[5]]] + data[6]))
        print(('Frustum angle: ', dataset.frustum_angle_list[i]))
        median_list.append(np.median(data[0][:, 0]))
        print((data[2], dataset.box3d_list[i], median_list[-1]))
        box3d_from_label = get_3d_box(class2size(data[5], data[6]), class2angle(data[3], data[4], 12), data[2])
        ps = data[0]
        seg = data[1]
        fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4), fgcolor=None, engine=None, size=(1000, 500))
        mlab.points3d(ps[:, 0], ps[:, 1], ps[:, 2], seg, mode='point', colormap='gnuplot', scale_factor=1, figure=fig)
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2, figure=fig)
        draw_gt_boxes3d([box3d_from_label], fig, color=(1, 0, 0))
        mlab.orientation_axes()
        raw_input()
    print(np.mean(np.abs(median_list)))