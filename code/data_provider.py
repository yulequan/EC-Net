import Queue
import os
import threading
import h5py
import matplotlib
import numpy as np


NUM_EDGE = 120
NUM_FACE = 800


def get_inverse_index(num):
    idx = np.random.permutation(num)
    new_idx = np.zeros(num).astype(np.int64)
    for id,item in enumerate(idx):
        new_idx[item]=id
    return idx, new_idx

def normalize_point_cloud(input):
    if len(input.shape)==2:
        axis = 0
    elif len(input.shape)==3:
        axis = 1
    centroid = np.mean(input, axis=axis, keepdims=True)
    input = input - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(input ** 2, axis=-1)),axis=axis,keepdims=True)
    input = input / furthest_distance
    return input, centroid,furthest_distance

def load_patch_data(skip_rate = 1):
    h5_filename = '../h5data/mix_CAD1k_halfnoise.h5'
    f = h5py.File(h5_filename)
    input = f['mc8k_input'][:]
    dist = f['mc8k_dist'][:]
    edge = f['edge'][:]
    edgepoint = f['edge_points'][:]
    face = f['face'][:]
    assert edge.shape[1]==NUM_EDGE
    assert face.shape[1]==NUM_FACE
    edge = np.reshape(edge,(-1,NUM_EDGE*2,3))
    face = np.reshape(face,(-1,NUM_FACE*3,3))
    edge = np.concatenate([edge,face,edgepoint],axis=1)
    name = f['name'][:]
    assert len(input) == len(edge)

    # ####
    h5_filename = '../h5data/mix_Virtualscan1k_halfnoise.h5'
    f = h5py.File(h5_filename)
    input1 = f['mc8k_input'][:]
    dist1 = f['mc8k_dist'][:]
    edge1 = f['edge'][:]
    edgepoint1 = f['edge_points'][:]
    face1 = f['face'][:]
    assert edge1.shape[1] == NUM_EDGE
    assert face1.shape[1] == NUM_FACE
    edge1 = np.reshape(edge1, (-1, NUM_EDGE * 2, 3))
    face1 = np.reshape(face1, (-1, NUM_FACE * 3, 3))
    edge1 = np.concatenate([edge1, face1, edgepoint1], axis=1)
    name1 = f['name'][:]
    assert len(input1) == len(edge1)
    input = np.concatenate([input,input1],axis=0)
    dist  = np.concatenate([dist,dist1],axis=0)
    edge  = np.concatenate([edge,edge1],axis=0)
    name = np.concatenate([name,name1])
    # ######

    data_radius = np.ones(shape=(len(input)))
    centroid = np.mean(input[:,:,0:3], axis=1, keepdims=True)
    input[:,:,0:3] = input[:,:,0:3] - centroid
    distance = np.sqrt(np.sum(input[:,:,0:3] ** 2, axis=-1))
    furthest_distance = np.amax(distance,axis=1,keepdims=True)
    input[:, :, 0:3] = input[:,:,0:3] / np.expand_dims(furthest_distance,axis=-1)
    dist = dist/furthest_distance

    edge[:, :, 0:3] = edge[:, :, 0:3] - centroid
    edge[:, :, 0:3] = edge[:, :, 0:3] / np.expand_dims(furthest_distance,axis=-1)

    input = input[::skip_rate]
    dist = dist[::skip_rate]
    edge = edge[::skip_rate]
    name = name[::skip_rate]
    data_radius = data_radius[::skip_rate]

    object_name = list(set([item.split('/')[-1].split('_')[0] for item in name]))
    object_name.sort()
    print "load object names {}".format(object_name)
    print "total %d samples" % (len(input))
    return input, dist, edge, data_radius, name


def rotate_point_cloud_and_gt(batch_data,batch_gt=None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    for k in range(batch_data.shape[0]):
        angles = np.random.uniform(size=(3)) * 2 * np.pi
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))


        batch_data[k, ..., 0:3] = np.dot(batch_data[k, ..., 0:3].reshape((-1, 3)), rotation_matrix)
        if batch_data.shape[-1]>5:
            batch_data[k, ..., 3:6] = np.dot(batch_data[k, ..., 3:6].reshape((-1, 3)), rotation_matrix)

        if batch_gt is not None:
            batch_gt[k, ..., 0:3]   = np.dot(batch_gt[k, ..., 0:3].reshape((-1, 3)), rotation_matrix)
            if batch_gt.shape[-1] > 5:
                batch_gt[k, ..., 3:6] = np.dot(batch_gt[k, ..., 3:6].reshape((-1, 3)), rotation_matrix)

    return batch_data,batch_gt


def shift_point_cloud_and_gt(batch_data, batch_gt = None, shift_range=0.3):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,0:3] += shifts[batch_index,0:3]

    if batch_gt is not None:
        for batch_index in range(B):
            batch_gt[batch_index, :, 0:3] += shifts[batch_index, 0:3]

    return batch_data,batch_gt


def random_scale_point_cloud_and_gt(batch_data, batch_gt = None, scale_low=0.5, scale_high=2):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,0:3] *= scales[batch_index]

    if batch_gt is not None:
        for batch_index in range(B):
            batch_gt[batch_index, :, 0:3] *= scales[batch_index]

    return batch_data,batch_gt,scales


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.03, angle_clip=0.09):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    for k in xrange(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        batch_data[k, ...,0:3] = np.dot(batch_data[k, ...,0:3].reshape((-1, 3)), R)
        # if batch_data.shape[-1]>3:
        #     batch_data[k, ..., 3:6] = np.dot(batch_data[k, ..., 3:6].reshape((-1, 3)), R)

    return batch_data


def jitter_perturbation_point_cloud(batch_data, sigma=0.005, clip=0.02):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data[:,:,3:] = 0
    jittered_data += batch_data
    return jittered_data

def convert_dist2rgba(dist,scale=10):
    cmap = matplotlib.cm.get_cmap('plasma')
    dist = dist * scale
    dist[dist > 1.0] = 1.0
    dist[dist < 0.0] = 0.0
    rgba = [cmap(item) for item in dist]
    rgba = np.asarray(rgba)
    rgba = (rgba * 255).astype(np.uint8)
    return rgba


def save_xyz(path, data):
    if not os.path.exists(os.path.split(path)[0]):
        os.makedirs(os.path.split(path)[0])
    np.savetxt(path, data, fmt='%.6f')
    # if np.random.rand()>1.9:
    #     show3d.showpoints(data[:, 0:3])

def save_ply(path, data):
    if not os.path.exists(os.path.split(path)[0]):
        os.makedirs(os.path.split(path)[0])
    point_num = data.shape[0]
    dimension = data.shape[1]
    with open(path,"w") as myfile:
        header =  "ply\nformat ascii 1.0\ncomment VCGLIB generated\nelement vertex %d\n"%(data.shape[0])
        header += "property float x\nproperty float y\nproperty float z\n"
        if dimension>3:
            header += "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n"
        if dimension>6:
            header += "property float quality\n"
        header += "element face 0\nproperty list uchar int vertex_indices\nend_header\n"
        myfile.write(header)

        for j in range(point_num):
            myfile.write("%f %f %f " % (data[j, 0], data[j, 1], data[j, 2]))
            myfile.write("%d %d %d %d " % (data[j, 3], data[j, 4], data[j, 5], data[j, 6]))
            myfile.write("%f\n" % (data[j, 7]))

    return None

def nonuniform_sampling(num = 4096, sample_num = 1024):
    sample = set()
    loc = np.random.rand()*0.8+0.1
    while(len(sample)<sample_num):
        a = int(np.random.normal(loc=loc,scale=0.3)*num)
        if a<0 or a>=num:
            continue
        sample.add(a)

    return list(sample)

class Fetcher(threading.Thread):
    def __init__(self, batch_size, num_point):
        super(Fetcher,self).__init__()
        self.queue = Queue.Queue(40)
        self.stopped = False
        self.batch_size = batch_size
        self.num_point = num_point
        input, dist, edge, data_radius, name = load_patch_data(skip_rate=1)

        # shuffle the order of points
        self.point_order = np.zeros([input.shape[0], input.shape[1]],np.int32)
        # for i in xrange(input.shape[0]):
        #     idx = np.random.permutation(input.shape[1])
        #     input[i] = input[i][idx]
        #     dist[i] = dist[i][idx]
        #     new_idx = np.zeros((input.shape[1]))
        #     for index,item in enumerate(idx):
        #         new_idx[item]=index
        #     self.point_order[i]= new_idx
        # self.point_order = self.point_order

        self.input_data = input
        self.dist_data = dist
        self.edge_data = edge
        self.radius_data = data_radius

        self.sample_cnt = self.input_data.shape[0]
        self.num_batches = self.sample_cnt//self.batch_size
        print "NUM_BATCH is %s"%(self.num_batches)

    def run(self):
        while not self.stopped:
            idx = np.arange(self.sample_cnt)
            np.random.shuffle(idx)
            self.input_data = self.input_data[idx, ...]
            self.dist_data = self.dist_data[idx, ...]
            self.edge_data = self.edge_data[idx, ...]
            self.radius_data = self.radius_data[idx, ...]
            self.point_order = self.point_order[idx,...]

            for batch_idx in range(self.num_batches):
                if self.stopped:
                    return None
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx + 1) * self.batch_size
                batch_data_input = self.input_data[start_idx:end_idx, :, :].copy()
                batch_data_dist = self.dist_data[start_idx:end_idx, :].copy()
                batch_data_edge = self.edge_data[start_idx:end_idx, :, :].copy()
                radius = self.radius_data[start_idx:end_idx].copy()
                point_order = self.point_order[start_idx:end_idx].copy()
                point_order0 = np.tile(np.reshape(np.arange(self.batch_size),(self.batch_size,1)),(1,point_order.shape[1]))
                point_order = np.stack([point_order0,point_order],axis=-1)
                # #shuffle the order of points
                for i in xrange(batch_data_input.shape[0]):
                    idx,new_idx = get_inverse_index(batch_data_input.shape[1])
                    batch_data_input[i]=batch_data_input[i][idx]
                    batch_data_dist[i]= batch_data_dist[i][idx]
                    point_order[i,:,1]=new_idx

                #data augmentation(rotate,scale, shift)
                batch_data_input, batch_data_edge = rotate_point_cloud_and_gt(batch_data_input,batch_data_edge)
                batch_data_input, batch_data_edge, scales = random_scale_point_cloud_and_gt(batch_data_input, batch_data_edge,
                                                                                        scale_low=0.8, scale_high=1.2)
                radius = radius*scales
                batch_data_dist = batch_data_dist*np.expand_dims(scales,axis=-1) #scale dist
                batch_data_input, batch_data_edge = shift_point_cloud_and_gt(batch_data_input, batch_data_edge,shift_range=0.2)
                batch_data_clean = batch_data_input.copy()

                batch_data_input = jitter_perturbation_point_cloud(batch_data_input, sigma=0.005, clip=0.015)
                batch_data_input = rotate_perturbation_point_cloud(batch_data_input, angle_sigma=0.03, angle_clip=0.09)

                self.queue.put((batch_data_input, batch_data_clean, batch_data_dist, batch_data_edge,radius,point_order))
        return None
    def fetch(self):
        if self.stopped:
            return None
        return self.queue.get()

    def shutdown(self):
        self.stopped = True
        print "Shutdown ....."
        while not self.queue.empty():
            self.queue.get()
        print "Remove all queue data"
