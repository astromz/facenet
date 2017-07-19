"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
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

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import random
from time import sleep
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#from PIL import Image, ImageFont, ImageDraw, ImageEnhance


def main(args):
    sleep(random.random())
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    dataset = facenet.get_dataset(args.input_dir)
    
    print('Creating networks and loading parameters')
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    
    minsize = 20 #20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
    
    if args.plot_bbox is True:
        labeled_img_dir = os.path.join(output_dir, 'bbox/')
        if not os.path.exists(labeled_img_dir):
            os.makedirs(labeled_img_dir)
    
    
    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned_imgs = 0

        if args.random_order:
            random.shuffle(dataset)
        for cls in dataset:
            output_class_dir = os.path.join(output_dir, cls.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
                if args.random_order:
                    random.shuffle(cls.image_paths)
            for image_path in cls.image_paths:
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                
                output_dir_individuaL_img = os.path.join(output_class_dir, filename)

                if not os.path.exists(output_dir_individuaL_img):
                    os.makedirs(output_dir_individuaL_img)    
                    print('\nProcessing image: {}'.format(image_path))
            
                    try:
                        img = misc.imread(image_path)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim<2:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_dir_individuaL_img))
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                        img = img[:,:,0:3]
    
                        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                        nrof_faces = bounding_boxes.shape[0]
                        print('Number of faces found = {}'.format(nrof_faces))
                        
                        if nrof_faces>0:
                            img_size = np.asarray(img.shape)[0:2]
                            
                            #if nrof_faces>1:
                            #    bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                            #    img_center = img_size / 2
                            #    offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                            #    offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                            #    index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                            #    det = det[index,:]
                                
                            # extract all found faces to small images    
                            nrof_successfully_aligned_faces = 0
                            bb = np.zeros((nrof_faces, 4), dtype=np.int32)
                            for i in range(nrof_faces):
                                outfile_name = os.path.join(output_dir_individuaL_img, '{}.png'.format(i))
                                if not os.path.exists(outfile_name):
                                    det = bounding_boxes[i,0:4]
                                    bb[i, 0] = np.maximum(det[0]-args.margin/2, 0)
                                    bb[i, 1] = np.maximum(det[1]-args.margin/2, 0)
                                    bb[i, 2] = np.minimum(det[2]+args.margin/2, img_size[1])
                                    bb[i, 3] = np.minimum(det[3]+args.margin/2, img_size[0])
                                    cropped = img[bb[i, 1]:bb[i, 3],bb[i, 0]:bb[i, 2],:]
                                    scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bicubic')
                                    misc.imsave(outfile_name, scaled)
                                    text_file.write('%s %d %d %d %d\n' % (outfile_name, bb[i,0], bb[i,1], bb[i,2], bb[i,3]))
                                    nrof_successfully_aligned_faces += 1
                            
                            # plot bboxes on original image and save it
                            if args.plot_bbox is True:
                                #import pdb;pdb.set_trace()
                                #print(img.max(), img.dtype)
                                '''
                                pil_img = Image.fromarray(img.astype(np.uint8))
                                draw = ImageDraw.Draw(pil_img)
                                for i in range(nrof_faces):
                                    draw.rectangle(bb[i,:], outline=(0,255,0), fill='white')
                                    font = ImageFont.truetype("sans-serif.ttf", 16)
                                    draw.text((bb[i,0]+3, bb[i,3]+3), "{}".format(i), font=font)
                                pil_img.save(labeled_img_dir + filename + '.png')
                                '''
                                fig, ax = plt.subplots(1)
                                ax.imshow(img, aspect='equal', shape=img_size)
                                rectangles = [patches.Rectangle(bb[i,[0,1]], bb[i,2]-bb[i,0],bb[i,3]-bb[i,1], linewidth=1,edgecolor='lawngreen',facecolor='none') for i in range(nrof_faces)]
                                [ax.add_patch(rect) for rect in rectangles]   # Add the patch to the Axes
                                ax.axis('off')
                                fig.savefig(labeled_img_dir + filename + '.png', bbox_inches='tight')
                                
                                print('saved image: {}'.format(labeled_img_dir + filename + '.png'))
                                
                            print('Number of successfully aligned faces : {}'.format(nrof_successfully_aligned_faces))
                            nrof_successfully_aligned_imgs += 1
                              
                        else:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_dir_individuaL_img))
                                
    print('\nTotal number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d \n' % nrof_successfully_aligned_imgs)
            

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order', 
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--plot_bbox', default=True, action="store_true",
        help='Save original images with labeled boundboxes')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
