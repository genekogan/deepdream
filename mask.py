from __future__ import print_function
import math
from random import random
import os
from os import listdir
from os.path import isfile, join
import copy
import itertools
from enum import Enum
from io import BytesIO
from IPython.display import clear_output, Image, display, HTML
import PIL.Image
import numpy as np
import scipy.misc
import sklearn.cluster
import cv2
from canvas import *
from util import *
#from lapnorm import *


default_color_labels = [[255,0,0],[0,255,255],[0,255,0],[255,0,255],[0,0,255],[255,255,0],[0,0,0],[255,255,255],[64,0,192],[192,64,0],[0,192,64],[192,0,64],[64,192,0],[0,64,192]]


def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 1)*255)
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

def crop_to_aspect_ratio(img, ar):
    ih, iw = img.shape[0:2]
    ar_img = float(iw)/ih
    if ar_img > ar:
        iw2 = ih * ar
        ix = (iw-iw2)/2
        img = img[:,int(ix):int(ix+iw2)]
    elif ar_img < ar:
        ih2 = float(iw) / ar
        iy = (ih-ih2)/2
        img = img[int(iy):int(iy+ih2),:]
    return img

def get_mask_sizes(init_size, oct_n, oct_s):
    sizes = [ np.int32(np.float32(init_size)) ]
    for octave in range(oct_n-1):
        hw = np.float32(sizes[-1]) / oct_s
        sizes.append(np.int32(hw))
    sizes = list(reversed(sizes))
    return sizes

def draw_mask(mask, flatten_blend=False, draw_rgb=False, animate=False, color_labels=None):
    color_labels = color_labels or default_color_labels
    h, w, nc = mask.shape
    mask_arr = np.zeros((h, w * nc))
    for c in range(nc):
        mask_arr[:, c * w:(c+1)*w] = mask[:, :, c]
    if flatten_blend:
        mask_arr = 0.5*(mask_arr>0.0)+0.5*(mask_arr==1.0)
    if draw_rgb:    
        mask_sum = np.sum(mask, axis=2)
        mask_norm = mask / mask_sum[:, :, np.newaxis] 
        mask_rgb = np.array(np.sum([[mask_norm[:, :, c] * clr for clr in color_labels[c%len(color_labels)]] for c in range(nc)], axis=0)).transpose((1,2,0))/255.
        showarray(mask_rgb)
    else:
        showarray(mask if draw_rgb else mask_arr)
    if animate:
        time.sleep(0.1)
        clear_output()



def mask_arcs(h, w, n, ctr_y, ctr_x, rad, period, t, blend=0.0, inwards=False, reverse=False):
    radius = rad * n
    mask = np.zeros((h, w, n))
    pts = np.array([[[i/(h-1.0),j/(w-1.0)] for j in range(w)] for i in range(h)])
    ctr = np.array([[[ctr_y, ctr_x] for j in range(w)] for i in range(h)])
    pts -= ctr
    dist = (pts[:,:,0]**2 + pts[:,:,1]**2)**0.5
    pct = (float(-t if inwards else period+t) / (n * period)) % 1.0
    d = (dist + radius * (1.0 - pct)) % radius
    for c in range(0, n):
        cidx = n-c-1 if (reverse != inwards) else c
        x1, x2 = rad * (n-c-1), rad * (n-c)
        x1b, x2b = x1 - d, d - x2
        dm = np.maximum(0, np.maximum(d-x2, x1-d)) 
        mask[:, :, cidx] = np.clip(1.0-x1b/(blend*rad), 0, 1)*np.clip(1.0-x2b/(blend*rad), 0, 1) if blend > 0 else (np.maximum(0, np.maximum(d-x2, x1-d)) <=0) * (dist < rad)
    return mask

def mask_rects(h, w, n, p1, p2, width, period, t, blend=0.0, reverse=False):
    mask = np.zeros((h, w, n))
    length = ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)**0.5
    m1 = 1e8 if p2[0]==p1[0] else (p2[1] - p1[1]) / (p2[0]-p1[0]) 
    b1 = p1[1] - m1 * p1[0]
    m2 = -1.0 / (m1+1e-8) #9e8 if m1==0 else (1e-8 if m1==9e8 else -1.0 / m1)
    pts = np.array([[[i/(h-1.0),j/(w-1.0)] for j in range(w)] for i in range(h)])
    x1, y1 = pts[:,:,0], pts[:,:,1]
    x_int = (y1 - m2 * x1 - b1) / (m1 - m2)
    y_int = m2 * x_int + y1 - m2 * x1
    isect = np.zeros(pts.shape)
    isect[:,:,0] = (y1 - m2 * x1 - b1) / (m1 - m2)
    isect[:,:,1] = m2 * isect[:,:,0] + y1 - m2 * x1
    inside = (isect[:,:,0] >= min(p1[0], p2[0])) * (isect[:,:,0] <= max(p1[0], p2[0])) 
    dist = ((isect[:,:,0]-p1[0])**2 + (isect[:,:,1]-p1[1])**2)**0.5 
    pts_from_isect = pts - isect
    dst_from_isect = ((pts_from_isect[:,:,0])**2 + (pts_from_isect[:,:,1])**2)**0.5 
    offset = length - length * float(t)/(n*period)
    dist = (dist + offset) % length
    dist_diag = (dist * inside) / length
    rad = 1.0 / n
    for r in range(n):
        ridx = n-r-1 if reverse else r
        t1, t2 = rad * (n-r-1), rad * (n-r)
        t1d, t2d = t1 - dist_diag, dist_diag - t2
        val = np.clip(1.0-t1d/(blend*rad), 0, 1)*np.clip(1.0-t2d/(blend*rad), 0, 1) if blend > 0 else (dist_diag >= t1)*(dist_diag<t2)
        dc = dst_from_isect - width/2.0
        val *= np.clip(1.0-dc/(blend*width), 0, 1) #(dst_from_isect <= width/2.0)
        mask[:, :, ridx] = val
    return mask

def mask_solid(h, w, n):
    mask = np.ones((h, w, n))
    return mask

def mask_interpolation(h, w, n, period, t, blend=0.0, reverse=False, cross_fade=False):
    mask = np.zeros((h, w, n))
    idx1 = int(math.floor(t / period) % n) if period > 0 else 0
    idx2 = int((idx1 + 1) % n)
    if reverse:
        idx1 = n-idx1-1
        idx2 = n-idx2-1
    pct = float(t % period) / period if period > 0 else 0
    progress = min(1.0, float(1.0 - pct) / blend) if blend > 0 else 0
    t2 = 1.0 - progress * progress
    t1 = 1.0 - t2 if cross_fade else 1.0
    mask[:, :, idx1] = t1
    mask[:, :, idx2] = t2
    return mask

def mask_image_manual(h, w, n, path, thresholds, blur_k, n_dilations):
    if len(thresholds) != n:
        raise ValueError('Number of thresholds doesn\'t match number of channels in mask')
    mask = np.zeros((h, w, n))
    img = cv2.imread(path, 0)
    img = crop_to_aspect_ratio(img, float(w)/h)
    #img = cv2.blur(img, (blur_k, blur_k))
    cumulative = np.zeros(img.shape[0:2]).astype('uint8')
    for channel, thresh in enumerate(thresholds):
        ret, img1 = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY_INV)
        img1 -= cumulative
        cumulative += img1
        for d in range(n_dilations):
            img1 = cv2.dilate(img1, (2, 2))    
        ih, iw = img.shape
        img1 = cv2.blur(img1, (blur_k, blur_k))
        img1 = cv2.resize(img1, (w, h))
        mask[:,:,channel] += img1/255.
    return mask

def mask_image_auto(h, w, n, path, blur_k, n_dilations):
    mask = np.zeros((h, w, n))
    img = cv2.imread(path, 0)
    img = crop_to_aspect_ratio(img, float(w)/h)
    #img = cv2.blur(img, (blur_k, blur_k))
    mask_cumulative = 255 * img.shape[0] * img.shape[1] / len(ch)
    cumulative = np.zeros(img.shape[0:2]).astype('uint8')
    thresh, thresholds = 0, []
    for channel in range(n):
        amt_mask = 0
        while amt_mask < mask_cumulative:
            thresh += 1
            ret, img1 = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY_INV)
            img1 -= cumulative
            amt_mask = np.sum(img1)
        cumulative += img1
        img1 = cv2.blur(img1, (blur_k, blur_k))
        img1 = cv2.resize(img1, (w, h))
        thresholds.append(thresh)
        mask[:,:,channel] += img1/255.
    return mask

def mask_image_kmeans(h, w, n, path, blur_k, n_dilations, prev_assign=None):
    mask = np.zeros((h, w, n))
    img = cv2.imread(path)
    img = cv2.blur(img, (blur_k, blur_k))
    img = crop_to_aspect_ratio(img, float(w)/h)
    img = cv2.resize(img, (w, h), cv2.INTER_NEAREST)   # CHANGE
    pixels = np.array(list(img)).reshape(h * w, 3)
    clusters, assign, _ = sklearn.cluster.k_means(pixels, n, init='k-means++', random_state=3425)
    #clusters = [[c[2],c[1],c[0]] for c in clusters]
    #new_pixels = np.array([clusters[a] for a in assign]).reshape((ih, iw, ic))
    if prev_assign is not None:
        assign_candidates, best_total = list(itertools.permutations(range(n))), -1
        for ac in assign_candidates:
            reassign = np.array([ac[a] for a in assign])
            total = np.sum(reassign == prev_assign)
            if total > best_total:
                best_total, best_assign = total, reassign
        assign = best_assign
    else:
        amts = [np.sum(assign==c) for c in range(n)]
        order = list(reversed(sorted(range(len(amts)), key=lambda k: amts[k])))
        reorder = [order.index(i) for i in range(n)]
        assign = np.array([reorder[a] for a in assign])
    for c in range(n):
        channel_mask = np.multiply(np.ones((h*w)), assign==c).reshape((h,w))
        for d in range(n_dilations):
            channel_mask = cv2.dilate(channel_mask, (3, 3))    
        #channel_mask = cv2.resize(channel_mask, (w, h), cv2.INTER_LINEAR)   # CHANGE
        mask[:,:,c] = channel_mask
    sums = [np.sum(mask[:,:,c]) for c in range(n)]
    return mask, assign

def mask_movie(h, w, n, path, thresholds, blur_k, n_dilations, t, idx1=None, idx2=None):
    if len(thresholds) != n:
        raise ValueError('Number of thresholds doesn\'t match number of channels in mask')
    frames = sorted([join(path,f) for f in listdir(path) if isfile(join(path, f))])
    if idx1 != None and idx2 != None:
        frames = frames[idx1:idx2]
    idx = t % len(frames)
    return mask_image_manual(h, w, n, frames[idx], thresholds, blur_k, n_dilations) 

def get_mask(mask, sy, sx, t, meta):
    m = mask
    if m['type'] == 'solid':
        m['n'] = 1
        masks = mask_solid(sy, sx, 1)
    elif m['type'] == 'interpolation':
        masks = mask_interpolation(sy, sx, n=m['n'], period=m['period'], t=t, blend=m['blend'], reverse=m['reverse'], cross_fade=True)
    elif m['type'] == 'arcs':
        masks = mask_arcs(sy, sx, n=m['n'], ctr_y=m['ctr_y'], ctr_x=m['ctr_x'], rad=m['radius'], period=m['period'], t=t, blend=m['blend'], inwards=m['inwards'], reverse=m['reverse'])
    elif m['type'] == 'rects':
        masks = mask_rects(sy, sx, n=m['n'], p1=m['p1'], p2=m['p2'], width=m['width'], period=m['period'], t=t, blend=m['blend'], reverse=m['reverse'])
    elif m['type'] == 'image':
        #masks = mask_image_manual(sy, sx, n=m['n'], path=m['path'], thresholds=m['thresholds'], blur_k=m['blur_k'], n_dilations=m['n_dilations'])
        #masks = mask_image_auto(sy, sx, n=m['n'], path=m['path'], blur_k=m['blur_k'], n_dilations=m['n_dilations'])
        masks, meta = mask_image_kmeans(sy, sx, n=m['n'], path=m['path'], blur_k=m['blur_k'], n_dilations=m['n_dilations'], prev_assign=meta)
    elif m['type'] == 'movie':
        masks = mask_movie(sy, sx, n=m['n'], path=m['path'], thresholds=m['thresholds'], blur_k=m['blur_k'], n_dilations=m['n_dilations'], t=t, idx1=m['idx1'], idx2=m['idx2'])
    if m['normalize']:
        mask_sum = np.maximum(np.ones(masks.shape[0:2]), np.sum(masks, axis=2))
        masks = masks / mask_sum[:, :, np.newaxis] 
    return masks, meta


def inject_image(img0, path, amt, matchHist=True):
    hist0 = get_histogram(img0.astype('uint8'), bright=False)
    img1 = scipy.misc.imread(path, mode='RGB')
    (h, w), (ih, iw) = (img0.shape[0:2]), (img1.shape[0:2])
    if float(w)/h > float(iw)/ih:
        d = ih - iw * float(h) / w
        if d>0:
            img1 = img1[int(d/2):-int(d/2),:,:]
    else:
        d = iw - ih * float(w) / h
        if d>0:
            img1 = img1[:, int(d/2):-int(d/2),:]
    img1 = resize(img1, (h, w))
    img2 = (1.0 - amt) * img0 + amt * img1
    if matchHist:
        img2 = match_histogram(img2, hist0)
    return img2.astype('float32')    

