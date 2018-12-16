#!/usr/bin/env python
# -*- coding: utf-8 -*-

import caffe
import cv2
import numpy as np
import os
import numpy.random as npr
import copy

def bbreg(boundingbox, reg):
    reg = reg.T 
    
    # calibrate bouding boxes
    if reg.shape[1] == 1:
        print "reshape of reg"
        pass # reshape of reg
    w = boundingbox[:,2] - boundingbox[:,0] + 1
    h = boundingbox[:,3] - boundingbox[:,1] + 1

    bb0 = boundingbox[:,0] + reg[:,0]*w
    bb1 = boundingbox[:,1] + reg[:,1]*h
    bb2 = boundingbox[:,2] + reg[:,2]*w
    bb3 = boundingbox[:,3] + reg[:,3]*h
    
    boundingbox[:,0:4] = np.array([bb0, bb1, bb2, bb3]).T
    #print "bb", boundingbox
    return boundingbox


def pad(boxesA, w, h):
    boxes = boxesA.copy() # shit, value parameter!!!

    tmph = boxes[:,3] - boxes[:,1] + 1
    tmpw = boxes[:,2] - boxes[:,0] + 1
    numbox = boxes.shape[0]

    dx = np.ones(numbox)
    dy = np.ones(numbox)
    edx = tmpw 
    edy = tmph

    x = boxes[:,0:1][:,0]
    y = boxes[:,1:2][:,0]
    ex = boxes[:,2:3][:,0]
    ey = boxes[:,3:4][:,0]
   
   
    tmp = np.where(ex > w)[0]
    if tmp.shape[0] != 0:
        edx[tmp] = -ex[tmp] + w-1 + tmpw[tmp]
        ex[tmp] = w-1

    tmp = np.where(ey > h)[0]
    if tmp.shape[0] != 0:
        edy[tmp] = -ey[tmp] + h-1 + tmph[tmp]
        ey[tmp] = h-1

    tmp = np.where(x < 1)[0]
    if tmp.shape[0] != 0:
        dx[tmp] = 2 - x[tmp]
        x[tmp] = np.ones_like(x[tmp])

    tmp = np.where(y < 1)[0]
    if tmp.shape[0] != 0:
        dy[tmp] = 2 - y[tmp]
        y[tmp] = np.ones_like(y[tmp])
    
    # for python index from 0, while matlab from 1
    dy = np.maximum(0, dy-1)
    dx = np.maximum(0, dx-1)
    y = np.maximum(0, y-1)
    x = np.maximum(0, x-1)
    edy = np.maximum(0, edy-1)
    edx = np.maximum(0, edx-1)
    ey = np.maximum(0, ey-1)
    ex = np.maximum(0, ex-1)

    return [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]

def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr


def rerec(bboxA):
    # convert bboxA to square
    w = bboxA[:,2] - bboxA[:,0]
    h = bboxA[:,3] - bboxA[:,1]
    l = np.maximum(w,h).T

    bboxA[:,0] = bboxA[:,0] + w*0.5 - l*0.5
    bboxA[:,1] = bboxA[:,1] + h*0.5 - l*0.5 
    bboxA[:,2:4] = bboxA[:,0:2] + np.repeat([l], 2, axis = 0).T 
    return bboxA


def nms(boxes, threshold, type):
    """nms
    :boxes: [:,0:5]
    :threshold: 0.5 like
    :type: 'Min' or others
    :returns: TODO
    """
    if boxes.shape[0] == 0:
        return np.array([])
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = np.array(s.argsort()) # read s using I
    
    pick = [];
    while len(I) > 0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if type == 'Min':
            o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
        else:
            o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where( o <= threshold)[0]]
    return pick


def generateBoundingBox(map, reg, scale, t):
    stride = 2
    cellsize = 12
    map = map.T
    dx1 = reg[0,:,:].T
    dy1 = reg[1,:,:].T
    dx2 = reg[2,:,:].T
    dy2 = reg[3,:,:].T
    (x, y) = np.where(map >= t)

    yy = y
    xx = x


    score = map[x,y]
    reg = np.array([dx1[x,y], dy1[x,y], dx2[x,y], dy2[x,y]])

    if reg.shape[0] == 0:
        pass
    boundingbox = np.array([yy, xx]).T

    bb1 = np.fix((stride * (boundingbox) + 1) / scale).T # matlab index from 1, so with "boundingbox-1"
    bb2 = np.fix((stride * (boundingbox) + cellsize - 1 + 1) / scale).T # while python don't have to
    score = np.array([score])

    boundingbox_out = np.concatenate((bb1, bb2, score, reg), axis=0)

    return boundingbox_out.T



def drawBoxes(im, boxes):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    for i in range(x1.shape[0]):
        cv2.rectangle(im, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), (0,255,0), 1)
    return im

def drawlandmark(im, points):
    for i in range(points.shape[0]):
        for j in range(5):
            cv2.circle(im, (int(points[i][j]), int(points[i][j+5])), 2, (255,0,0))
    return im


from time import time
_tstart_stack = []
def tic():
    _tstart_stack.append(time())
def toc(fmt="Elapsed: %s s"):
    print fmt % (time()-_tstart_stack.pop())


def detect_face(img, minsize, PNet, RNet, ONet, threshold, fastresize, factor):
    
    img2 = img.copy()

    factor_count = 0
    total_boxes = np.zeros((0,9), np.float)
    points = []
    h = img.shape[0]
    w = img.shape[1]
    minl = min(h, w)
    img = img.astype(float)
    m = 12.0/minsize
    minl = minl*m
    

    # create scale pyramid
    scales = []
    while minl >= 12:
        scales.append(m * pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    
    # first stage
    for scale in scales:
        hs = int(np.ceil(h*scale))
        ws = int(np.ceil(w*scale))

        if fastresize:
            im_data = (img-127.5)*0.0078125 # [0,255] -> [-1,1]
            im_data = cv2.resize(im_data, (ws,hs)) # default is bilinear
        else: 
            im_data = cv2.resize(img, (ws,hs)) # default is bilinear
            im_data = (im_data-127.5)*0.0078125 # [0,255] -> [-1,1]
        #im_data = imResample(img, hs, ws); print "scale:", scale


        im_data = np.swapaxes(im_data, 0, 2)
        im_data = np.array([im_data], dtype = np.float)
        PNet.blobs['data'].reshape(1, 3, ws, hs)
        PNet.blobs['data'].data[...] = im_data
        out = PNet.forward()
    
        boxes = generateBoundingBox(out['prob1'][0,1,:,:], out['conv4-2'][0], scale, threshold[0])
        if boxes.shape[0] != 0:
            pick = nms(boxes, 0.5, 'Union')
            if len(pick) > 0 :
                boxes = boxes[pick, :]

        if boxes.shape[0] != 0:
            total_boxes = np.concatenate((total_boxes, boxes), axis=0)
         
    #np.save('total_boxes_101.npy', total_boxes)

    #####
    # 1 #
    #####
    # print "[1]:",total_boxes.shape[0]
    #print total_boxes
    #return total_boxes, [] 


    numbox = total_boxes.shape[0]
    if numbox > 0:
        # nms
        pick = nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
        # print "[2]:",total_boxes.shape[0]
        
        # revise and convert to square
        regh = total_boxes[:,3] - total_boxes[:,1]
        regw = total_boxes[:,2] - total_boxes[:,0]
        t1 = total_boxes[:,0] + total_boxes[:,5]*regw
        t2 = total_boxes[:,1] + total_boxes[:,6]*regh
        t3 = total_boxes[:,2] + total_boxes[:,7]*regw
        t4 = total_boxes[:,3] + total_boxes[:,8]*regh
        t5 = total_boxes[:,4]
        total_boxes = np.array([t1,t2,t3,t4,t5]).T
        total_boxes = rerec(total_boxes) # convert box to square
        # print "[4]:",total_boxes.shape[0]
        
        total_boxes[:,0:4] = np.fix(total_boxes[:,0:4])
        # print "[4.5]:",total_boxes.shape[0]
        #print total_boxes
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)


    numbox = total_boxes.shape[0]
    if numbox > 0:
        # second stage

        # construct input for RNet
        tempimg = np.zeros((numbox, 24, 24, 3)) # (24, 24, 3, numbox)
        for k in range(numbox):
            tmp = np.zeros((int(tmph[k]) +1, int(tmpw[k]) + 1,3))
            tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]
            #print "y,ey,x,ex", y[k], ey[k], x[k], ex[k]
            #print "tmp", tmp.shape
            
            tempimg[k,:,:,:] = cv2.resize(tmp, (24, 24))

        #print tempimg.shape
        #print tempimg[0,0,0,:]
        tempimg = (tempimg-127.5)*0.0078125 # done in imResample function wrapped by python

        # RNet
        tempimg = np.swapaxes(tempimg, 1, 3)
        #print tempimg[0,:,0,0]
        
        RNet.blobs['data'].reshape(numbox, 3, 24, 24)
        RNet.blobs['data'].data[...] = tempimg
        out = RNet.forward()

        score = out['prob1'][:,1]
        #print 'score', score
        pass_t = np.where(score>threshold[1])[0]
        #print 'pass_t', pass_t
        
        score =  np.array([score[pass_t]]).T
        total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis = 1)
        # print "[5]:",total_boxes.shape[0]
        #print total_boxes
        #print "1.5:",total_boxes.shape
        
        mv = out['conv5-2'][pass_t, :].T
        #print "mv", mv
        if total_boxes.shape[0] > 0:
              # print "[6]:", total_boxes.shape[0]
              total_boxes = bbreg(total_boxes, mv[:, :])
              # print "[7]:", total_boxes.shape[0]
              total_boxes = rerec(total_boxes)
              # print "[8]:", total_boxes.shape[0]

    return total_boxes

def main():
    img_dir = "../../data/WIDER_train/images/"
    imglistfile = "../../data/wider_face_train.txt"
    with open(imglistfile, 'r') as f:
        annotations = f.readlines()
    num = len(annotations)
    print "%d pics in total" % num

    neg_save_dir = "./48/negative/"
    pos_save_dir = "./48/positive/"
    part_save_dir = "./48/part/"
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)
    if not os.path.exists(part_save_dir):
        os.mkdir(part_save_dir)
    image_size = 48
    f1 = open('./48/pos_48.txt', 'w')
    f2 = open('./48/neg_48.txt', 'w')
    f3 = open('./48/part_48.txt', 'w')

    p_idx = 0  # positive
    n_idx = 0  # negative
    d_idx = 0  # dont care
    image_idx = 0

    minsize = 20
    caffe_model_path = "../model"
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    
    caffe.set_mode_gpu()
    PNet = caffe.Net(caffe_model_path+"/det1.prototxt", caffe_model_path+"/det1.caffemodel", caffe.TEST)
    RNet = caffe.Net(caffe_model_path+"/det2.prototxt", caffe_model_path+"/det2.caffemodel", caffe.TEST)
    ONet = caffe.Net(caffe_model_path+"/det3.prototxt", caffe_model_path+"/det3.caffemodel", caffe.TEST)


    for annotation in annotations:
        # imgpath = imgpath.split('\n')[0]
        annotation = annotation.strip().split(' ')
        bbox = map(float, annotation[1:])
        gts = np.array(bbox, dtype=np.float32).reshape(-1, 4)
        img_path = img_dir + annotation[0]+".jpg"

        #print "######\n", img_path
        if image_idx % 100 == 0:
            print "(%s products = negative %s + positive %s + part %s) %s / %s" % (p_idx + n_idx + d_idx, n_idx, p_idx, d_idx, image_idx, num)
        image_idx += 1
        img = cv2.imread(img_path)
        img_matlab = img.copy()
        h = img.shape[0]
        w = img.shape[1]
        tmp = img_matlab[:,:,2].copy()
        img_matlab[:,:,2] = img_matlab[:,:,0]
        img_matlab[:,:,0] = tmp

        boundingboxes = detect_face(img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)

        #img = drawBoxes(img, boundingboxes)
        #cv2.imshow('img', img)
        # cv2.waitKey(1000)

        # generate positive,negative,part samples
        for box in boundingboxes:
            x_left, y_top, x_right, y_bottom, _ = box
            crop_w = x_right - x_left + 1
            crop_h = y_bottom - y_top + 1
            # ignore box that is too small or beyond image border
            #if crop_w < image_size / 2 or crop_h < image_size / 2:
                #continue
            if x_left < 0 or y_top < 0:
                continue

            # compute intersection over union(IoU) between current box and all gt boxes
            Iou = IoU(box, gts)
            cropped_im = img[int(y_top):int(y_bottom + 1) , int(x_left):int(x_right + 1)]
            resized_im = cv2.resize(cropped_im, (image_size, image_size))
            #try:
            #    resized_im = cv2.resize(cropped_im, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            #except  Exception as e:
            #    print " 1 "
            #    print e

            # save negative images and write label
            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write("%s/negative/%s.jpg" % (image_size, n_idx) + ' 0')
                f2.write(" -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1\n")
                cv2.imwrite(save_file, resized_im)
                n_idx += 1

                # jingxiang
                iLR = copy.deepcopy(resized_im)
                resized_h = resized_im.shape[0]
                resized_w = resized_im.shape[1]
                for i in range(resized_h):
                    for j in range(resized_w):
                        iLR[i,resized_w-1-j]=resized_im[i,j]
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write("%s/negative/%s.jpg" % (image_size, n_idx) + ' 0')
                f2.write(" -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1\n")
                cv2.imwrite(save_file, iLR)
                n_idx += 1

            else:
                # find gt_box with the highest iou
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(crop_w)
                offset_y1 = (y1 - y_top) / float(crop_h)
                # offset_x2 = (x2 - x_left) / float(crop_w)
                # offset_y2 = (y2 - y_top) / float(crop_h)
                offset_x2 = (x2 - x_right)  / float(crop_w)
                offset_y2 = (y2 - y_bottom )/ float(crop_h)

                # save positive and part-face images and write labels
                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    f1.write("%s/positive/%s.jpg" % (image_size, p_idx) + ' 1 %.6f %.6f %.6f %.6f' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    f1.write(" -1 -1 -1 -1 -1 -1 -1 -1 -1 -1\n")
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                    # jingxiang
                    iLR = copy.deepcopy(resized_im)
                    resized_h = resized_im.shape[0]
                    resized_w = resized_im.shape[1]
                    for i in range(resized_h):
                        for j in range(resized_w):
                            iLR[i,resized_w-1-j]=resized_im[i,j]
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    f1.write("48/positive/%s.jpg"%p_idx + ' 1 %.6f %.6f %.6f %.6f -1 -1 -1 -1 -1 -1 -1 -1 -1 -1\n'%(-offset_x2, offset_y1, -offset_x1, offset_y2))
                    cv2.imwrite(save_file, iLR)
                    p_idx += 1

                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    f3.write("%s/part/%s.jpg" % (image_size, d_idx) + ' -1 %.6f %.6f %.6f %.6f' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    f3.write(" -1 -1 -1 -1 -1 -1 -1 -1 -1 -1\n")
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1

                    # jingxiang
                    iLR = copy.deepcopy(resized_im)
                    resized_h = resized_im.shape[0]
                    resized_w = resized_im.shape[1]
                    for i in range(resized_h):
                        for j in range(resized_w):
                            iLR[i,resized_w-1-j]=resized_im[i,j]
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    f3.write("48/part/%s.jpg"%d_idx + ' -1 %.6f %.6f %.6f %.6f -1 -1 -1 -1 -1 -1 -1 -1 -1 -1\n'%(-offset_x2, offset_y1, -offset_x1, offset_y2))
                    cv2.imwrite(save_file, iLR)
                    d_idx += 1


    f.close()
    f1.close()
    f2.close()
    f3.close()

if __name__ == "__main__":
    main()
