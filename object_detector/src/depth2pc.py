#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2


class depth2pc():

    def __init__(self, rgb_image, depth_image, fx, fy, cx, cy, scalingfactor):
        self.rgb = rgb_image
        self.depth = depth_image
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scalingfactor = scalingfactor
        self.height = self.rgb.shape[0]
        self.width = self.rgb.shape[1]

    def point_cloud_generator(self):
        depth = np.asarray(self.depth)
        self.Z = depth / self.scalingfactor
        X = np.zeros((self.height, self.width))
        Y = np.zeros((self.height, self.width))
        for i in range(self.width):
            X[:, i] = np.full(X.shape[0], i)

        self.X = ((X - self.cx) * self.Z) / self.fx
        for i in range(self.height):
            Y[i, :] = np.full(Y.shape[1], i)
        self.Y = ((Y - self.cy) * self.Z) / self.fy

        df=np.zeros((6,self.height*self.width))
        df[0] = self.X.T.reshape(-1)
        df[1] = self.Y.T.reshape(-1)
        df[2] = self.Z.T.reshape(-1)
        img = np.array(self.rgb)
        df[3] = img[:, :, 0:1].reshape(-1)
        df[4] = img[:, :, 1:2].reshape(-1)
        df[5] = img[:, :, 2:3].reshape(-1)

        return df.T




if __name__ == '__main__':
    try:
        depth_image = np.loadtxt('/home/he/Kit/Masterarbeit/3D_Mapping_ws/src/object_detector/photos/depth_1.txt')
        rgb_image = cv2.imread('/home/he/Kit/Masterarbeit/3D_Mapping_ws/src/object_detector/photos/rgb_1.png')
        # cv2.imshow("Image window", rgb_img)
        # cv2.waitKey(0)
        print(depth_image.shape)
        print(rgb_image.shape)
        a = depth2pc(rgb_image, depth_image, 979, 979, 1027, 770, 1)
        pc = a.point_cloud_generator()
        print(type(pc))
    except KeyboardInterrupt:
        print ("Shutting down depth2pc node.")
        #cv2.destroyAllWindows()
