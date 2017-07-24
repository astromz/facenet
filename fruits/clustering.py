"""A modified clustering module that clusters and groups pre-cropped faces into 
different folder for easy labeling.
"""
# MIT License
# 
# Copyright (c) 2017 Ming Zhao@ NYTimes
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import shutil
import sys
sys.path.append('/Users/207229/Google Drive/Work/facial/facenet/src/')

import time
import math
#import pickle
#from sklearn.svm import SVC
from sklearn.cluster import DBSCAN, AffinityPropagation
from sklearn.metrics.pairwise import euclidean_distances
import facenet
from glob import glob
from matplotlib import pyplot as plt


#%% 
def group_faces_to_dir(all_img_paths, labels, cluster_path):
    '''copy all face images to their corresponding dirs determined by clustering algo.
    '''
    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)
    for l in labels:
        label_dir = cluster_path + '/{}'.format(l)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        # copy face images to their corresponding label dir
        pic_ind = np.where(labels == l)[0]
        [shutil.copy(all_img_paths[i], os.path.join(label_dir, all_img_paths[i].split('/')[-1])) for i in pic_ind]
    return

#%% ########                            
def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set

#%%
def main(args):

    clustering_method = args.clustering_method
    tag = args.tag
    
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            np.random.seed(seed=args.seed)
    
            all_img_paths = glob(args.data_dir + '*/*.png', recursive=True)
            assert len(all_img_paths) > 0, 'Number of images must be >0!'
            print('\nFound {} images in total'.format(len(all_img_paths)))
            
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(all_img_paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            print('Number of batches = {}'.format(nrof_batches_per_epoch))
            #nrof_batches_per_epoch = 1
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                print('Working on batch #{}'.format(i))
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = all_img_paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            print('Completed generating embeddings\n')
    # End of TF graph
        
    #%%    
    
    if clustering_method == 'AffinityPropagation':
    
        # Use AffinityPropagation
        affinity_matrix = -euclidean_distances(emb_array, squared=True)
        af = AffinityPropagation(preference = np.min(affinity_matrix)).fit(emb_array)
        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_
        n_clusters_ = len(cluster_centers_indices)
        print('Estimated number of clusters: %d' % n_clusters_)
    
        cluster_dir_name = 'AP_pref_{0:3.3f}_{}'.format(np.min(affinity_matrix,tag))
        
        
    elif clustering_method == 'dbscan':
        # Pre-calculate affinity matrix, using squared_L2 distance (required for FaceNet representation)
        affinity_matrix = euclidean_distances(emb_array, squared=True)
        eps_arr = np.linspace(0.4, 1.0, 25)
        
        n_clusters_arr = []
        n_outliers_arr = []
        for eps in eps_arr:
            #db = DBSCAN(eps=eps, min_samples=3,metric='euclidean').fit(emb_array)
    
            db = DBSCAN(eps=eps, min_samples=3,metric='precomputed').fit(affinity_matrix)
    
            labels = db.labels_
    
            # Number of clusters in labels, ignoring noise if present.
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_outliers = np.sum(labels == -1)
            n_clusters_arr.append(n_clusters)
            n_outliers_arr.append(n_outliers)
        
        print('---> Please check the plots and find the best eps value !!!')
        plt.ion()
        plt.figure()
        plt.plot(eps_arr, n_clusters_arr, 'o-')
        plt.show()
        
        plt.figure()
        plt.plot(eps_arr, n_outliers_arr, 'o-')
        plt.show()
    
    
    #%% After exploring the parameters we need to hand pick a best esp value for the final, actual clustering
        best_eps_input = input('Enter your pick for the best EPS value (default=0.625): ')
        if best_eps_input =='':
            best_eps = 0.625
        else:
            best_eps = float(best_eps_input)
        print(' best_eps value for final iteration = {}'.format(best_eps))

        db = DBSCAN(eps=best_eps, min_samples=3,metric='precomputed').fit(affinity_matrix)
    
        labels = db.labels_
    
        # Number of clusters in labels, ignoring noise if present.
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = np.sum(labels == -1)
        print('N_clusters, N_outliers = {}, {}'.format(n_clusters, n_outliers))
        
        cluster_dir_name = 'DBSCAN_eps_{0:2.3f}_{1}'.format(best_eps, tag)
        # Write model to pkl file
    #%%
    if args.create_clustered_dir is True:
        if args.output_data_dir == '':
            data_path = os.path.dirname(os.path.abspath(args.data_dir))
        else:
            data_path = args.output_data_dir
        print('Clutered data are saved at: {}'.format(os.path.join(data_path, cluster_dir_name)) )
        group_faces_to_dir(all_img_paths, labels, os.path.join(data_path, cluster_dir_name) )
    print('Clustering completed! Congrats!\n')
    return



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing aligned face patches.')
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=100)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=42)
    parser.add_argument('--clustering_method', type=str,
        help='Algorithm used for clustering', default='dbscan')
    parser.add_argument('--tag', type=str,
        help='Unique tag name for the instance', default=time.strftime("%Y%b%d_%H%M"))
    parser.add_argument('--create_clustered_dir', default=False, action="store_true",
        help='Set to True to actually create the clustered directory (safeguard)')
    parser.add_argument('--output_data_dir', default='',
        help='Dir path to store clustered images.')

    args = parser.parse_args()
    
    #%%
    """ # uncomment this block for interactive ipython debugging and testing
    arg_str = '../../data/nyt/test_aligned_60/raw2/ ../../models/facenet_models/20170512-110547/20170512-110547.pb --batch_size 100'
    args = parser.parse_args(arg_str.split())    
    clustering_method = 'dbscan'
    """
    
    main(args)
