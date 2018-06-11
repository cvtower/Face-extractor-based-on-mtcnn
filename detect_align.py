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

#
# Borrowed from davidsandberg's facenet project: https://github.com/davidsandberg/facenet
# From this directory:
#   facenet/src/align
#
# Just keep the MTCNN related stuff and removed other codes
# python package required:
#     tensorflow, opencv,numpy


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import detect_face
import cv2
import math
from skimage import transform as trans

###not for insight repo.2018/5/1
def faceAlign_v0(input_image, points, output_size = (96, 96), ec_mc_y = 40):
    '''
    return the size-fixed align image with the given facial landmarks 
    '''
    allAlignFaces = []
    print(points)

    for i in range(points.shape[0]):
        '''
        points examples, which is different with happynear's examples
        [[219.68787 281.90543 244.70497 239.8125  293.89606]
         [155.76303 135.53867 185.15842 228.07518 212.40016]]
        '''
        if points.shape[0] == 10:
            currentPoints = points.reshape(2,5)
            # print currentPoints
        
            eye_center = ((currentPoints[0][0] + currentPoints[0][1]) / 2, (currentPoints[1][0] + currentPoints[1][1]) / 2)
            mouth_center = ((currentPoints[0][3] + currentPoints[0][4]) / 2, (currentPoints[1][3] + currentPoints[1][4]) / 2)
            angle = math.atan2(mouth_center[0] - eye_center[0], mouth_center[1] - eye_center[1]) / math.pi * -180.0
            scale = ec_mc_y / math.sqrt((mouth_center[0] - eye_center[0])**2 + (mouth_center[1] - eye_center[1])**2)
            center = ((currentPoints[0][0] + currentPoints[0][1] + currentPoints[0][3] + currentPoints[0][4]) / 4, 
                      (currentPoints[1][0] + currentPoints[1][1] + currentPoints[1][3] + currentPoints[1][4]) / 4)
            rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
            rot_mat[0][2] -= (center[0] - output_size[0] / 2)
            rot_mat[1][2] -= (center[1] - output_size[1] / 2)
            warp_dst = cv2.warpAffine(input_image, rot_mat, output_size)
            allAlignFaces.append(warp_dst)
        else:
            # only store the images with 10 facial lamdmarks
            continue

    return allAlignFaces
###works for insight repo.2018/5/1
def faceAlign_v1(img, src, points):
    image_size = (112, 112)
    tform = trans.SimilarityTransform()
    pset_x = points[:5]
    pset_y = points[5:]
    dst = np.array(list(zip(pset_x, pset_y))).astype(np.float32).reshape(5,2)
    tform.estimate(dst,src)
    M = tform.params[:2,:]
    warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
    return warped

def main(args):
    
    sess = tf.Session()
    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    filename =args.input 
    output_filename =args.output

    draw = cv2.imread(filename)
    draw = cv2.cvtColor(draw,cv2.COLOR_BGR2RGB)
    #cv2.imshow('img',img)
    #cv2.waitKey(0)

    bounding_boxes, points = detect_face.detect_face(draw, minsize, pnet, rnet, onet, threshold, factor)

    nrof_faces = bounding_boxes.shape[0]

    for b in bounding_boxes:
        cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0))
        print(b)

    cv2.imwrite(output_filename,draw)
    #print(len(points))

    img2 = draw.copy()

    x_ = [38.2946, 73.5318, 56.0252, 41.5493, 70.7299]
    y_ = [51.6963, 51.5014, 71.7366, 92.3655, 92.2041]
    src = np.array(list(zip(x_,y_))).astype(np.float32).reshape(5,2)
    if(bounding_boxes.shape[0] == 0):
        alignfaces = []
    else:
        v0_face = faceAlign_v0(img2, points)
        cv2.imwrite('v0_face.jpg',v0_face[0])
        v1_face = faceAlign_v1(img2, src, points)
        cv2.imwrite('v1_face.jpg',v1_face)

    print('Total %d face(s) detected, view info in %s' % (nrof_faces,output_filename))
            

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='image to be detected for faces.',default='./test.jpg')
    parser.add_argument('--output', type=str, help='new image with boxed faces',default='output.jpg')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
