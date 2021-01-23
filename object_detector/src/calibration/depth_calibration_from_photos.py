import os
import numpy as np
import cv2
import glob


def calibration(inter_corner_shape, size_per_grid, img_dir,img_type):
    # criteria: only for subpix calibration, which is not used here.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    w,h = inter_corner_shape
    # cp_int: corner point in int form, save the coordinate of corner points in world sapce in 'int' form
    # like (0,0,0), (1,0,0), (2,0,0) ....,(10,7,0).
    cp_int = np.zeros((w*h,3), np.float32)
    cp_int[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
    # cp_world: corner point in world space, save the coordinate of corner points in world space.
    cp_world = cp_int*size_per_grid
    
    obj_points = [] # the points in world space
    img_points = [] # the points in image space (relevant to obj_points)
    images = glob.glob(img_dir + '**.' + img_type)
    count = 0
    shape = []
    for fname in images:
        img = cv2.imread(fname)
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # find the corners, cp_img: corner points in pixel space.
        ret, cp_img = cv2.findChessboardCorners(gray_img, (w,h), None)
        if shape == []:
            shape = (img.shape[1],img.shape[0])
        # if ret is True, save.
        if ret == True:
            cv2.cornerSubPix(gray_img,cp_img,(11,11),(-1,-1),criteria)
            obj_points.append(cp_world)
            img_points.append(cp_img)
            print (fname)
            print (count)
            count += 1
            # print 'image:%d'%count
            # view the corners
            cv2.drawChessboardCorners(img, (w,h), cp_img, ret)
            cv2.imshow('FoundCorners',img)
            cv2.imwrite('../../photos/calibration_depth/find_corners/photo_'+str(count)+'.png',img)
            cv2.waitKey(1)
    cv2.destroyAllWindows()
    # calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, shape, None, None, flags=cv2.CALIB_RATIONAL_MODEL)
    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, shape, None, None)
    # print 'end calibration'
    # print 'ret:'
    # print (ret)
    # print 'camera matrix:'
    # print (mtx)
    # print 'distortion coefficients:'
    # print (dist)
    parameter = [mtx[0][0],mtx[0][2],mtx[1][1],mtx[1,2]] + dist[0][:8].tolist()
    print 'parameter:'
    print (parameter)
    np.savetxt("../../../Azure_Kinect_ROS_Driver/calibration/depth_camera_calibration.txt", parameter,fmt='%f',delimiter=' ')
    print 'save parameter'
    mean_error = 0
    for i in range(len(obj_points)):
        img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
        mean_error += error
    print "total error: ", mean_error / len(obj_points)
    
    return mtx,dist
    
if __name__ == '__main__':
    inter_corner_shape = (15,10)
    size_per_grid = 0.072
    img_dir = "../../photos/calibration_depth/raw_img/"
    img_type = "png"
    calibration(inter_corner_shape, size_per_grid, img_dir,img_type)