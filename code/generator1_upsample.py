import tensorflow as tf
from utils import tf_util
from utils.pointnet_util import pointnet_sa_module, pointnet_fp_module


def get_gen_model(point_cloud, is_training, scope, bradius = 1.0, reuse=None,use_bn = False,use_ibn = False,
                  use_normal=False,bn_decay=None, up_ratio = 4,num_addpoint=600,idx=None,is_crop=False):

    print "Crop flag is ",is_crop

    with tf.variable_scope(scope,reuse=reuse) as sc:
        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[1].value
        l0_xyz = point_cloud[:,:,0:3]
        if use_normal:
            l0_points = point_cloud[:,:,3:]
        else:
            l0_points = None
        # Layer 1
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=num_point, radius=bradius*0.1,bn=use_bn,ibn = use_ibn,
                                                           nsample=12, mlp=[32, 32, 64], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer1')

        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=num_point/2, radius=bradius*0.2,bn=use_bn,ibn = use_ibn,
                                                           nsample=32, mlp=[64, 64, 128], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer2')

        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=num_point/4, radius=bradius*0.4,bn=use_bn,ibn = use_ibn,
                                                           nsample=32, mlp=[128, 128, 256], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer3')

        l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=num_point/8, radius=bradius*0.6,bn=use_bn,ibn = use_ibn,
                                                           nsample=32, mlp=[256, 256, 512], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer4')

        # Feature Propagation layers
        if not is_training:
            l0_xyz = tf.gather_nd(l0_xyz, idx[:, :int(num_point * 1/2), :])
            l1_points = tf.gather_nd(l1_points, idx[:, :int(num_point * 1/2), :])
        elif is_crop:
            l0_xyz = tf.gather_nd(l0_xyz, idx[:, :int(num_point * 1/2), :])
            l1_points = tf.gather_nd(l1_points, idx[:, :int(num_point * 1/2), :])


        up_l4_points = pointnet_fp_module(l0_xyz, l4_xyz, None, l4_points, [64], is_training, bn_decay,
                                       scope='fa_layer1',bn=use_bn,ibn = use_ibn)

        up_l3_points = pointnet_fp_module(l0_xyz, l3_xyz, None, l3_points, [64], is_training, bn_decay,
                                       scope='fa_layer2',bn=use_bn,ibn = use_ibn)

        up_l2_points = pointnet_fp_module(l0_xyz, l2_xyz, None, l2_points, [64], is_training, bn_decay,
                                       scope='fa_layer3',bn=use_bn,ibn = use_ibn)

        feat = tf.concat([up_l4_points, up_l3_points, up_l2_points, l1_points], axis=-1)
        feat = tf.expand_dims(feat, axis=2)

        #branch1: the new generate points
        with tf.variable_scope('up_layer', reuse=reuse):
            up_feat_list = []
            for i in range(up_ratio):
                up_feat = tf_util.conv2d(feat, 256, [1, 1],
                                         padding='VALID', stride=[1, 1],
                                         bn=False, is_training=is_training,
                                         scope='conv1_%d' % (i), bn_decay=bn_decay)

                up_feat = tf_util.conv2d(up_feat, 128, [1, 1],
                                         padding='VALID', stride=[1, 1],
                                         bn=use_bn, is_training=is_training,
                                         scope='conv2_%d' % (i),
                                         bn_decay=bn_decay)
                up_feat_list.append(up_feat)
        up_feat = tf.concat(up_feat_list, axis=1)
        dist_feat = tf_util.conv2d(up_feat, 64, [1, 1],
                                   padding='VALID', stride=[1, 1],
                                   bn=False, is_training=is_training,
                                   scope='dist_fc1', bn_decay=bn_decay, weight_decay=0.0)
        dist = tf_util.conv2d(dist_feat, 1, [1, 1],
                              padding='VALID', stride=[1, 1],
                              bn=False, is_training=is_training,
                              scope='dist_fc2', bn_decay=bn_decay,
                              activation_fn=None, weight_decay=0.0)
        dist = tf.squeeze(dist, axis=[2, 3])

        #branch2: dist to the edge
        combined_feat = tf.concat((up_feat, dist_feat),axis=-1)
        coord_feat = tf_util.conv2d(combined_feat, 64, [1, 1],
                                    padding='VALID', stride=[1, 1],
                                    bn=False, is_training=is_training,
                                    scope='coord_fc1', bn_decay=bn_decay, weight_decay=0.0)
        r_coord = tf_util.conv2d(coord_feat, 3, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=False, is_training=is_training,
                                 scope='coord_fc2', bn_decay=bn_decay,
                                 activation_fn=None, weight_decay=0.0)
        coord = tf.squeeze(r_coord, [2]) + tf.tile(l0_xyz[:, :, 0:3], (1, up_ratio, 1))

        # prune the points according to probability(how to better prune it? as a guidance???)
        # poolsize = int(num_addpoint * 1.2)
        # val,idx1 = tf.nn.top_k(-dist,poolsize)
        # tmp_idx0 = tf.tile(tf.reshape(tf.range(batch_size),(batch_size,1)),(1,num_addpoint))
        # tmp_idx1 = tf.random_uniform((batch_size,num_addpoint),0,poolsize,tf.int32)
        # idx1 = tf.gather_nd(idx1,tf.stack([tmp_idx0,tmp_idx1],axis=-1))
        edge_dist, idx1 = tf.nn.top_k(-dist, num_addpoint)
        idx0 = tf.tile(tf.reshape(tf.range(batch_size),(batch_size,1)),(1,num_addpoint))
        idx = tf.stack([idx0,idx1],axis=-1)

    return dist, coord, idx