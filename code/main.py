import argparse
import os
import socket
import sys
import time
from glob import glob

import numpy as np
import tensorflow as tf
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_provider
import model_utils
import generator1_upsample as MODEL_GEN
from data_provider import NUM_EDGE, NUM_FACE
from GKNN import GKNN
from tf_ops.sampling.tf_sampling import farthest_point_sample
from utils import pc_util

parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='test', help='train or test [default: train]')
parser.add_argument('--gpu', default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='../model/pretrain', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [1024/2048] [default: 1024]')
parser.add_argument('--up_ratio',  type=int,  default=4,   help='Upsampling Ratio [default: 4]')
parser.add_argument('--is_crop',type= bool, default=True, help='Use cropped points in training [default: True]')
parser.add_argument('--max_epoch', type=int, default=200, help='Epoch to run [default: 200]')
parser.add_argument('--batch_size', type=int, default=12, help='Batch Size during training [default: 12]')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--assign_model_path',default=None, help='Pre-trained model path [default: None]')
parser.add_argument('--use_uniformloss',type= bool, default=False, help='Use uniformloss [default: False]')

FLAGS = parser.parse_args()
print socket.gethostname()
print FLAGS

ASSIGN_MODEL_PATH=FLAGS.assign_model_path
USE_UNIFORM_LOSS = FLAGS.use_uniformloss
IS_CROP = FLAGS.is_crop
PHASE = FLAGS.phase
GPU_INDEX = FLAGS.gpu
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
#NUM_ADDPOINT = FLAGS.num_addpoint
if PHASE=='train':
    NUM_ADDPOINT=512
else:
    NUM_ADDPOINT=96
UP_RATIO = FLAGS.up_ratio
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
MODEL_DIR = FLAGS.log_dir
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_INDEX

def log_string(out_str):
    global LOG_FOUT
    LOG_FOUT.write(out_str)
    LOG_FOUT.flush()

class Network(object):
    def __init__(self):
        return

    def build_graph(self,is_training=True,scope='generator'):
        bn_decay = 0.95
        self.step = tf.Variable(0, trainable=False)
        self.pointclouds_input = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3))
        self.pointclouds_radius = tf.placeholder(tf.float32, shape=(BATCH_SIZE))
        self.pointclouds_idx = tf.placeholder(tf.int32,shape=(BATCH_SIZE,NUM_POINT,2))
        self.pointclouds_edge = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_EDGE, 6))
        self.pointclouds_surface = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_FACE, 9))

        # create the generator model
        self.pred_dist, self.pred_coord,self.idx = MODEL_GEN.get_gen_model(self.pointclouds_input, is_training, scope=scope, bradius=1.0,
                                                    num_addpoint=NUM_ADDPOINT,reuse=None, use_normal=False, use_bn=False,use_ibn=False,
                                                    bn_decay=bn_decay, up_ratio=UP_RATIO,idx=self.pointclouds_idx,is_crop=IS_CROP)

        # calculate the ground truth point-to-edge distances for upsampled_points
        self.gt_dist = model_utils.distance_point2edge(self.pred_coord, self.pointclouds_edge)
        self.gt_dist = tf.sqrt(tf.reduce_min(self.gt_dist, axis=-1))
        self.gt_dist_truncated = tf.minimum(0.5, self.gt_dist)
        self.pred_dist = tf.minimum(0.5,tf.maximum(0.0,self.pred_dist))




        if is_training is False:
            # identify the edge points in the inference phase
            # We first filter the potential edge points with the top-k (NUM_ADDPOINT) minimum predicted point-to-edge distances.
            self.pred_edgecoord = tf.gather_nd(self.pred_coord, self.idx)
            self.pred_edgedist = tf.gather_nd(self.pred_dist, self.idx)

            # The following code is okay when the batch size is 1.
            self.edge_threshold = tf.constant(0.05, tf.float32, [1])
            indics = tf.where(tf.less_equal(self.pred_edgedist, self.edge_threshold))  # (?,2)
            self.select_pred_edgecoord = tf.gather_nd(self.pred_edgecoord, indics)  # (?,3)
            self.select_pred_edgedist = tf.gather_nd(self.pred_edgedist, indics)  # (?,3)
            return

        # Loss1: Edge distance regressionn loss
        self.dist_mseloss = (self.gt_dist_truncated - self.pred_dist) ** 2
        self.dist_mseloss = 5 * tf.reduce_mean(self.dist_mseloss / tf.expand_dims(self.pointclouds_radius ** 2, axis=-1))
        tf.summary.histogram('dist/gt', self.gt_dist_truncated)
        tf.summary.histogram('dist/pred', self.pred_dist)
        tf.summary.scalar('loss/dist_loss', self.dist_mseloss)


        # Loss2: Edge loss
        # We first filter the potential edge points with the top-k (NUM_ADDPOINT) minimum predicted point-to-edge distances.
        self.pred_edgecoord = tf.gather_nd(self.pred_coord, self.idx)
        self.pred_edgedist = tf.gather_nd(self.pred_dist, self.idx)
        self.gt_edgedist_truncated = tf.gather_nd(self.gt_dist_truncated, self.idx)

        # At the beginning of the training, the predicted point-to-edge distance is not accurate.
        # When identifying the edge points, we use the weighted sum of the predicted and ground truth point-to-edge distances.
        weight = tf.maximum(0.5 - tf.to_float(self.step) / 20000.0, 0.0)
        self.edgemask = tf.to_float(tf.less_equal(weight * self.gt_edgedist_truncated + (1 - weight) * self.pred_edgedist, 0.15))
        self.edge_loss = 50*tf.reduce_sum(self.edgemask * self.gt_edgedist_truncated ** 2 / tf.expand_dims(self.pointclouds_radius ** 2, axis=-1)) / (tf.reduce_sum(self.edgemask) + 1.0)
        tf.summary.scalar('weight',weight)
        tf.summary.histogram('dist/edge_dist', self.gt_edgedist_truncated)
        tf.summary.histogram('loss/edge_mask', self.edgemask)
        tf.summary.scalar('loss/edge_loss', self.edge_loss)

        # Loss3: Surface loss
        self.surface_dist = model_utils.distance_point2mesh(self.pred_coord, self.pointclouds_surface)
        self.surface_dist = tf.reduce_min(self.surface_dist, axis=2)
        self.surface_loss = 500*tf.reduce_mean(self.surface_dist / tf.expand_dims(self.pointclouds_radius ** 2, axis=-1))
        tf.summary.scalar('loss/plane_loss', self.surface_loss)

        # Loss4: Repulsion loss
        self.perulsionloss = 500*model_utils.get_perulsion_loss(self.pred_coord, numpoint=NUM_POINT * UP_RATIO)
        tf.summary.scalar('loss/perulsion_loss', self.perulsionloss)

        # Total loss
        self.total_loss = self.surface_loss + self.edge_loss + self.perulsionloss + self.dist_mseloss + tf.losses.get_regularization_loss()

        # make optimizer
        gen_update_ops = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if op.name.startswith(scope)]
        gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith(scope)]
        with tf.control_dependencies(gen_update_ops):
            self.gen_train = tf.train.AdamOptimizer(BASE_LEARNING_RATE, beta1=0.9).minimize(self.total_loss, var_list=gen_tvars,
                                                                                            colocate_gradients_with_ops=False,
                                                                                            global_step=self.step)
        # merge summary and add pointclouds summary
        tf.summary.scalar('loss/regularation', tf.losses.get_regularization_loss())
        tf.summary.scalar('loss/total_loss', self.total_loss)
        self.merged = tf.summary.merge_all()

        self.pointclouds_image_input = tf.placeholder(tf.float32, shape=[None, 500, 1500, 1])
        pointclouds_input_summary = tf.summary.image('1_input', self.pointclouds_image_input, max_outputs=1)
        self.pointclouds_image_pred = tf.placeholder(tf.float32, shape=[None, 500, 1500, 1])
        pointclouds_pred_summary = tf.summary.image('2_pred', self.pointclouds_image_pred, max_outputs=1)
        self.pointclouds_image_gt = tf.placeholder(tf.float32, shape=[None, 500, 1500, 1])
        pointclouds_gt_summary = tf.summary.image('3_edge', self.pointclouds_image_gt, max_outputs=1)
        self.image_merged = tf.summary.merge([pointclouds_input_summary, pointclouds_pred_summary, pointclouds_gt_summary])

    def train(self,assign_model_path=None):
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        # config.log_device_placement = False
        with tf.Session(config=config) as self.sess:
            self.train_writer = tf.summary.FileWriter(os.path.join(MODEL_DIR, 'train'), self.sess.graph)
            init = tf.global_variables_initializer()
            self.sess.run(init)

            # restore the model
            saver = tf.train.Saver(max_to_keep=10)
            restore_epoch, checkpoint_path = model_utils.pre_load_checkpoint(MODEL_DIR)
            global LOG_FOUT
            if restore_epoch == 0:
                LOG_FOUT = open(os.path.join(MODEL_DIR, 'log_train.txt'), 'w')
                LOG_FOUT.write(str(socket.gethostname()) + '\n')
                LOG_FOUT.write(str(FLAGS) + '\n')
            else:
                LOG_FOUT = open(os.path.join(MODEL_DIR, 'log_train.txt'), 'a')
                saver.restore(self.sess, checkpoint_path)

            ###assign the generator with another model file
            if assign_model_path is not None:
                print "Load pre-train model from %s" % (assign_model_path)
                assign_saver = tf.train.Saver(
                    var_list=[var for var in tf.trainable_variables() if var.name.startswith("generator")])
                assign_saver.restore(self.sess, assign_model_path)

            ##read data
            self.fetchworker = data_provider.Fetcher(BATCH_SIZE, NUM_POINT)
            self.fetchworker.start()
            for epoch in tqdm(range(restore_epoch, MAX_EPOCH + 1), ncols=45):
                log_string('**** EPOCH %03d ****\t' % (epoch))
                self.train_one_epoch()
                if epoch % 20 == 0:
                    saver.save(self.sess, os.path.join(MODEL_DIR, "model"), global_step=epoch)
            self.fetchworker.shutdown()

    def train_one_epoch(self):
        loss_sum = []
        fetch_time = 0
        for batch_idx in range(self.fetchworker.num_batches):
            start = time.time()
            batch_data_input, batch_data_clean, batch_data_dist, batch_data_edgeface, radius,point_order = self.fetchworker.fetch()
            batch_data_edge = np.reshape(batch_data_edgeface[:,0:2*NUM_EDGE,:],(BATCH_SIZE,NUM_EDGE,6))
            batch_data_face = np.reshape(batch_data_edgeface[:, 2*NUM_EDGE:2*NUM_EDGE+3*NUM_FACE,:],(BATCH_SIZE, NUM_FACE, 9))
            # A = batch_data_face[:,:,3:6]-batch_data_face[:,:,0:3]
            # B = batch_data_face[:,:,6:9]-batch_data_face[:,:,0:3]
            # batch_data_normal = np.cross(A,B)+1e-12
            # batch_data_normal = batch_data_normal / np.sqrt(np.sum(batch_data_normal ** 2, axis=-1, keepdims=True))
            # batch_data_edgepoint =batch_data_edgeface[:, 2*NUM_EDGE+3*NUM_FACE:, :]
            end = time.time()
            fetch_time += end - start

            feed_dict = {self.pointclouds_input: batch_data_input,
                         self.pointclouds_idx: point_order,
                         self.pointclouds_edge: batch_data_edge,
                         self.pointclouds_surface: batch_data_face,
                         self.pointclouds_radius: radius}
            _, summary, step, pred_coord, pred_edgecoord, edgemask, edge_loss = self.sess.run(
                [self.gen_train, self.merged, self.step, self.pred_coord, self.pred_edgecoord, self.edgemask, self.edge_loss], feed_dict=feed_dict)
            self.train_writer.add_summary(summary, step)
            loss_sum.append(edge_loss)
            edgemask[:,0:5]=1
            pred_edgecoord = pred_edgecoord[0][edgemask[0]==1]
            if step % 30 == 0:
                pointclouds_image_input = pc_util.point_cloud_three_views(batch_data_input[0, :, 0:3])
                pointclouds_image_input = np.expand_dims(np.expand_dims(pointclouds_image_input, axis=-1), axis=0)
                pointclouds_image_pred = pc_util.point_cloud_three_views(pred_coord[0, :, 0:3])
                pointclouds_image_pred = np.expand_dims(np.expand_dims(pointclouds_image_pred, axis=-1), axis=0)
                pointclouds_image_gt = pc_util.point_cloud_three_views(pred_edgecoord[:, 0:3])
                pointclouds_image_gt = np.expand_dims(np.expand_dims(pointclouds_image_gt, axis=-1), axis=0)
                feed_dict = {self.pointclouds_image_input: pointclouds_image_input,
                             self.pointclouds_image_pred: pointclouds_image_pred,
                             self.pointclouds_image_gt: pointclouds_image_gt}
                summary = self.sess.run(self.image_merged, feed_dict)
                self.train_writer.add_summary(summary, step)
            if step % 100 ==0:
                loss_sum = np.asarray(loss_sum)
                log_string('step: %d edge_loss: %f\n' % (step, round(loss_sum.mean(), 4)))
                print 'datatime:%s edge_loss:%f' % (round(fetch_time, 4), round(loss_sum.mean(), 4))
                loss_sum = []


    def patch_prediction(self, patch_point, sess, ratio, edge_threshold=0.05):
        #normalize the point clouds
        patch_point, centroid, furthest_distance = data_provider.normalize_point_cloud(patch_point)
        new_idx = np.stack((np.zeros((NUM_POINT)).astype(np.int64), np.arange(NUM_POINT)), axis=-1)

        pred, pred_edge, pred_edgedist = sess.run([self.pred_coord, self.select_pred_edgecoord, self.select_pred_edgedist],
                                                    feed_dict={self.pointclouds_input: np.expand_dims(patch_point,axis=0),
                                                               self.pointclouds_radius: np.ones(1),
                                                               self.edge_threshold: np.asarray([edge_threshold])/ratio,
                                                               self.pointclouds_idx: np.expand_dims(new_idx, axis=0)
                                                               })

        pred = np.squeeze(centroid + pred * furthest_distance, axis=0)
        pred_edge = centroid + pred_edge * furthest_distance
        pred_edgedist = pred_edgedist * furthest_distance
        return pred, pred_edge, pred_edgedist


    def pc_prediction(self, gm, sess, patch_num_ratio=3, edge_threshold=0.05):
        ## get patch seed from farthestsampling
        points = tf.convert_to_tensor(np.expand_dims(gm.data,axis=0),dtype=tf.float32)
        start= time.time()
        seed1_num = int(gm.data.shape[0] / (NUM_POINT/2) * patch_num_ratio)

        ## FPS sampling
        seed = farthest_point_sample(seed1_num*2, points).eval()[0]
        seed_list = seed[:seed1_num]
        print "farthest distance sampling cost", time.time() - start
        ratios = np.random.uniform(1.0,1.0,size=[seed1_num])

        input_list = []
        up_point_list=[]
        up_edge_list = []
        up_edgedist_list = []
        fail = 0
        for seed,ratio in tqdm(zip(seed_list,ratios)):
            try:
                patch_size = int(NUM_POINT * ratio)
                idx = np.asarray(gm.bfs_knn(seed,patch_size))
                # idx = np.asarray(gm.geodesic_knn(seed,patch_size))
                if len(idx)<NUM_POINT:
                    fail = fail + 1
                    continue
                idx1 = np.random.permutation(idx.shape[0])[:NUM_POINT]
                idx1.sort()
                idx = idx[idx1]
                point = gm.data[idx]
            except:
                fail= fail+1
                continue
            up_point,up_edgepoint,up_edgedist = self.patch_prediction(point, sess,ratio,edge_threshold)

            input_list.append(point)
            up_point_list.append(up_point)
            up_edge_list.append(up_edgepoint)
            up_edgedist_list.append(up_edgedist)
        print "total %d fails" % fail

        input = np.concatenate(input_list,axis=0)
        pred = np.concatenate(up_point_list,axis=0)

        pred_edge = np.concatenate(up_edge_list, axis=0)
        print "total %d edgepoint" % pred_edge.shape[0]
        pred_edgedist = np.concatenate(up_edgedist_list,axis=0)
        rgba = data_provider.convert_dist2rgba(pred_edgedist, scale=10)
        pred_edge = np.hstack((pred_edge, rgba, pred_edgedist.reshape(-1, 1)))

        return input, pred, pred_edge


    def test_hierarical_prediction(self, input_folder=None, save_path=None):
        self.saver = tf.train.Saver()
        _, restore_model_path = model_utils.pre_load_checkpoint(MODEL_DIR)
        print restore_model_path

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            self.saver.restore(sess, restore_model_path)
            total_time = 0
            samples = glob(input_folder)
            samples.sort()
            for point_path in samples:
                edge_path = None
                print point_path, edge_path
                start = time.time()
                gm = GKNN(point_path, edge_path, patch_size=NUM_POINT, patch_num=30,add_noise=False,normalization=True)

                ##get the edge information
                _,pred,pred_edge = self.pc_prediction(gm,sess,patch_num_ratio=3, edge_threshold=0.05)
                end = time.time()
                print "total time: ",end-start

                path = os.path.join(save_path, point_path.split('/')[-1][:-4] + "_input.xyz")
                data_provider.save_xyz(path, gm.data)

                path = os.path.join(save_path, point_path.split('/')[-1][:-4] + "_output.xyz")
                data_provider.save_xyz(path, pred)

                path = os.path.join(save_path, point_path.split('/')[-1][:-4] + "_outputedge.ply")
                data_provider.save_ply(path, pred_edge)

            print total_time/len(samples)


if __name__ == "__main__":
    np.random.seed(int(time.time()))
    tf.set_random_seed(int(time.time()))
    if PHASE=='train':
        assert not os.path.exists(os.path.join(MODEL_DIR, 'code/'))
        os.makedirs(os.path.join(MODEL_DIR, 'code/'))
        os.system('cp -r * %s' % (os.path.join(MODEL_DIR, 'code/')))  # bkp of model def
        network = Network()
        network.build_graph(is_training=True)
        network.train()
        LOG_FOUT.close()
    else:
        network = Network()
        BATCH_SIZE = 1
        NUM_EDGE = 1000
        network.build_graph(is_training=False)
        input_folder = '../eval_input/*.xyz'
        save_path = os.path.join('../eval_result/')
        network.test_hierarical_prediction(input_folder=input_folder, save_path=save_path)
