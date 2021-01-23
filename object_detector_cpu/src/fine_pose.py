import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import least_squares



class generate_station_model:

    def __init__(self, center, size, angle):
        self.center = center
        self.size = size
        self.angle = angle

    def generate_points(self):
        # cx = self.center[0]
        # cy = self.center[1]
        # cz = self.center[2]
        w = self.size[0]
        h = self.size[1]
        l = self.size[2]
        w_list = np.linspace(-w/2,w/2,50)
        h_list = np.linspace(-h/2,h/2,50)
        l_list = np.linspace(-l/2,l/2,50)

        w1 = -(w/2-0.115)
        w2 = w/2-0.115
        h1 = h/2-0.165
        h2 = -(h/2-0.09)
        h3 = h/2-0.1
        l1 = -(l/2-0.115)
        l2 = l/2-0.115

        x = np.zeros(5*50*50)
        y = np.zeros(5*50*50)
        z = np.zeros(5*50*50)
        x[0:50*50] = np.tile(w_list,50)
        y[0:50*50] = np.repeat(h_list,50)
        z[0:50*50] = -l/2
        mask1 = np.logical_not(((x[0:50*50]>w1) & (x[0:50*50]<w2)) & (((y[0:50*50]<h1) & (y[0:50*50]>h2)) | (y[0:50*50] > h3)))
        x[50*50:2*50*50] = np.tile(w_list,50)
        y[50*50:2*50*50] = np.repeat(h_list,50)
        z[50*50:2*50*50] = l/2
        mask2 = mask1
        x[2*50*50:3*50*50] = -w/2
        y[2*50*50:3*50*50] = np.repeat(h_list,50)
        z[2*50*50:3*50*50] = np.tile(l_list,50)
        mask3 = np.logical_not(((z[2*50*50:3*50*50]>l1) & (z[2*50*50:3*50*50]<l2)) & (((y[2*50*50:3*50*50]<h1) & (y[2*50*50:3*50*50]>h2)) | (y[2*50*50:3*50*50] > h3)))
        x[3*50*50:4*50*50] = w/2
        y[3*50*50:4*50*50] = np.repeat(h_list,50)
        z[3*50*50:4*50*50] = np.tile(l_list,50)
        mask4 = mask3
        x[4*50*50:5*50*50] = np.tile(w_list,50)
        y[4*50*50:5*50*50] = -h/2
        z[4*50*50:5*50*50] = np.repeat(l_list,50)
        mask5 = y[4*50*50:5*50*50]<0
        mask = np.hstack((mask1,mask2,mask3,mask4,mask5))
        points = np.zeros((5*50*50,3))
        points[:,0] = x
        points[:,1] = y
        points[:,2] = z
        points = points[mask]
        points = rotate_pc_along_y(points,self.angle)
        points = points + self.center.repeat(points.shape[0],0)
        return points

class generate_station_model_with_normal:

    def __init__(self, center, size, angle):
        self.center = center
        self.size = size
        self.angle = angle

    def generate_points(self):
        # cx = self.center[0]
        # cy = self.center[1]
        # cz = self.center[2]
        w = self.size[0]
        h = self.size[1]
        l = self.size[2]
        w_list = np.linspace(-w/2,w/2,50)
        h_list = np.linspace(-h/2,h/2,50)
        l_list = np.linspace(-l/2,l/2,50)

        w1 = -(w/2-0.115)
        w2 = w/2-0.115
        h1 = h/2-0.165
        h2 = -(h/2-0.09)
        h3 = h/2-0.1
        l1 = -(l/2-0.115)
        l2 = l/2-0.115

        x = np.zeros(5*50*50)
        y = np.zeros(5*50*50)
        z = np.zeros(5*50*50)
        n = np.zeros((5*50*50,3))
        x[0:50*50] = np.tile(w_list,50)
        y[0:50*50] = np.repeat(h_list,50)
        z[0:50*50] = -l/2
        n[0:50*50] = np.array([[0,0,1]]).repeat(50*50,0)
        mask1 = np.logical_not(((x[0:50*50]>w1) & (x[0:50*50]<w2)) & (((y[0:50*50]<h1) & (y[0:50*50]>h2)) | (y[0:50*50] > h3)))
        x[50*50:2*50*50] = np.tile(w_list,50)
        y[50*50:2*50*50] = np.repeat(h_list,50)
        z[50*50:2*50*50] = l/2
        n[50*50:2*50*50] = np.array([[0,0,-1]]).repeat(50*50,0)
        mask2 = mask1
        x[2*50*50:3*50*50] = -w/2
        y[2*50*50:3*50*50] = np.repeat(h_list,50)
        z[2*50*50:3*50*50] = np.tile(l_list,50)
        n[2*50*50:3*50*50] = np.array([[1,0,0]]).repeat(50*50,0)
        mask3 = np.logical_not(((z[2*50*50:3*50*50]>l1) & (z[2*50*50:3*50*50]<l2)) & (((y[2*50*50:3*50*50]<h1) & (y[2*50*50:3*50*50]>h2)) | (y[2*50*50:3*50*50] > h3)))
        x[3*50*50:4*50*50] = w/2
        y[3*50*50:4*50*50] = np.repeat(h_list,50)
        z[3*50*50:4*50*50] = np.tile(l_list,50)
        n[3*50*50:4*50*50] = np.array([[-1,0,0]]).repeat(50*50,0)
        mask4 = mask3
        x[4*50*50:5*50*50] = np.tile(w_list,50)
        y[4*50*50:5*50*50] = -h/2
        z[4*50*50:5*50*50] = np.repeat(l_list,50)
        n[4*50*50:5*50*50] = np.array([[0,1,0]]).repeat(50*50,0)
        mask5 = y[4*50*50:5*50*50]<0
        mask = np.hstack((mask1,mask2,mask3,mask4,mask5))
        points = np.zeros((5*50*50,6))
        points[:,0] = x
        points[:,1] = y
        points[:,2] = z
        points[:,3:] = n
        points = points[mask]
        points[:,:3] = rotate_pc_along_y(points[:,:3],self.angle)
        points[:,3:] = rotate_pc_along_y(points[:,3:],self.angle)
        points[:,:3] = points[:,:3] + self.center.repeat(points.shape[0],0)
        return points

class cicp:

    def __init__(self,A, B, max_iterations=20, tolerance_a=np.pi/200, tolerance_t=0.0001):
        self.A = A
        self.B = B
        self.max_iterations = max_iterations
        self.tolerance_a = tolerance_a
        self.tolerance_t = tolerance_t

    def cicp(self):

        m = self.A.shape[1]

        # make points homogeneous, copy them to maintain the originals
        src = np.ones((m+1,self.A.shape[0]))
        dst = np.ones((m+1,self.B.shape[0]))
        src[:m,:] = np.copy(self.A.T)
        dst[:m,:] = np.copy(self.B.T[:m,:])
        normal = np.copy(self.B.T[m:,:]) 

        # find the nearest neighbors between the current source and destination points
        indices = self.nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        x0 = np.array([0, 0, 0], dtype=float)
        result = least_squares(
            self.point_plane_residual,
            x0,
            jac='2-point',
            method='trf',
            loss='soft_l1',
            args=(src[:m,:].T, dst[:m,indices].T, normal[:,indices].T)
        )
    
        T = np.identity(4)
        for _ in range(self.max_iterations-1):
            x0 = result.x
            if (abs(x0[0])<self.tolerance_a) and (abs(x0[1])<self.tolerance_t) and (abs(x0[2])<self.tolerance_t):
                # print("R:{},t1:{},t2:{} ".format(x0[0],x0[1],x0[2]))
                # print('break')
                break
            T0 = self.transformation(x0,np.identity(4))
            T = self.transformation(x0,T)
            src = np.dot(T0, src)

            indices = self.nearest_neighbor(src[:m,:].T, dst[:m,:].T)
            result = least_squares(
                self.point_plane_residual,
                x0,
                jac='2-point',
                method='trf',
                loss='soft_l1',
                args=(src[:m,:].T, dst[:m,indices].T, normal[:,indices].T)
            )

        x = result.x
        T = self.transformation(x,T)
        R = T[:3, :3]
        t = T[:3, 3]
        return T,R,t

    def point_plane_residual(self,x,src,dst,normal):

        angle = x[0]
        t1 = x[1]
        t2 = x[2]

        cosval = np.cos(angle)
        sinval = np.sin(angle)
        transform = np.identity(4)
        transform[:3, :3] = np.array([[cosval, 0, -sinval], [0, 1, 0], [sinval, 1, cosval]])
        transform[:3, 3] = t1 *  np.array([1, 0, 0], dtype=float) + t2 *  np.array([0, 0, 1], dtype=float)

        position_residuals = np.dot(transform[:3, :3], src.T) + transform[:3, 3].reshape(3, 1) - dst.T
        residuals = np.abs(position_residuals * normal.T)
        residuals = np.sum(residuals, axis=0)

        return residuals

    def transformation(self,x0,base_transform):
        angle = x0[0]
        t1 = x0[1]
        t2 = x0[2]

        cosval = np.cos(angle)
        sinval = np.sin(angle)
        transform = np.identity(4)
        R = np.array([[cosval, 0, -sinval], [0, 1, 0], [sinval, 0, cosval]])
        t = t1 * np.array([1, 0, 0], dtype=float) + t2 * np.array([0, 0, 1], dtype=float)
        transform[:3, :3] = R
        transform[:3, 3] = t
        transform = np.dot(transform, base_transform)

        return transform

    def nearest_neighbor(self,src, dst):
        '''
        Find the nearest (Euclidean) neighbor in dst for each point in src
        Input:
            src: Nxm array of points
            dst: Nxm array of points
        Output:
            distances: Euclidean distances of the nearest neighbor
            indices: dst indices of the nearest neighbor
        '''

        # assert src.shape == dst.shape

        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)
        return indices.ravel()

def rotate_pc_along_y(pc,rot_angle):
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc