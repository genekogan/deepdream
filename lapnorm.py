from __future__ import print_function
from io import BytesIO
import math, time, copy, json, os
import glob
from os import listdir
from os.path import isfile, join
from random import random
from io import BytesIO
from enum import Enum
from functools import partial
import PIL.Image
from IPython.display import clear_output, Image, display, HTML
import numpy as np
import scipy.misc
import cv2
import tensorflow as tf
from util import *
from mask import get_mask_sizes, inject_image
from canvas import *
#from mask import crop_image, get_mask_sizes, inject_image


#Grab inception model from online and unzip it (you can skip this step if you've already downloaded the model.
#!wget -P ../data/ https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
#!unzip ../data/inception5h.zip -d ../data/inception5h/
#!rm ../data/inception5h.zip

#Create a session and load the Inception graph, then print the available layers.
model_fn = 'inception5h/tensorflow_inception_graph.pb'

graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

t_input = tf.placeholder(np.float32, name='input') # define the input tensor
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input':t_preprocessed})

layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]

print('Number of layers', len(layers))
print('Total number of feature channels:', sum(feature_nums))



def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 1)*255)
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

def visstd(a, s=0.1):
    '''Normalize the image range for visualization'''
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

def T(layer):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("import/%s:0"%layer)

def render_naive(t_obj, img0, iter_n=20, step=1.0):
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    img = img0.copy()
    for i in range(iter_n):
        g, score = sess.run([t_grad, t_score], {t_input:img})
        # normalizing the gradient, so the same step size should work 
        g /= g.std()+1e-8        # for different layers and networks
        img += g*step
    return img

def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.'''
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]

resize = tffunc(np.float32, np.int32)(resize)

def calc_grad_tiled(img, t_grad, tile_size=512):
    '''Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over 
    multiple iterations.'''
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in range(0, max(h-sz//2, sz),sz):
        for x in range(0, max(w-sz//2, sz),sz):
            sub = img_shift[y:y+sz,x:x+sz]
            g = sess.run(t_grad, {t_input:sub})
            grad[y:y+sz,x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

def render_multiscale(t_obj, img0, iter_n=10, step=1.0, octave_n=3, octave_scale=1.4):
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    img = img0.copy()
    for octave in range(octave_n):
        if octave>0:
            hw = np.float32(img.shape[:2])*octave_scale
            img = resize(img, np.int32(hw))
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            # normalizing the gradient, so the same step size should work 
            g /= g.std()+1e-8        # for different layers and networks
            img += g*step
            print('.', end='')
        print("octave %d/%d"%(octave+1, octave_n))
    #clear_output()
    return img

# Add code from the original notebook for cutting the high frequencies using Laplacian pyramids.

k = np.float32([1,4,6,4,1])
k = np.outer(k, k)
k5x5 = k[:,:,None,None]/k.sum()*np.eye(3, dtype=np.float32)

def lap_split(img):
    '''Split the image into lo and hi frequency components'''
    with tf.name_scope('split'):
        lo = tf.nn.conv2d(img, k5x5, [1,2,2,1], 'SAME')
        lo2 = tf.nn.conv2d_transpose(lo, k5x5*4, tf.shape(img), [1,2,2,1])
        hi = img-lo2
    return lo, hi

def lap_split_n(img, n):
    '''Build Laplacian pyramid with n splits'''
    levels = []
    for i in range(n):
        img, hi = lap_split(img)
        levels.append(hi)
    levels.append(img)
    return levels[::-1]

def lap_merge(levels):
    '''Merge Laplacian pyramid'''
    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            img = tf.nn.conv2d_transpose(img, k5x5*4, tf.shape(hi), [1,2,2,1]) + hi
    return img

def normalize_std(img, eps=1e-10):
    '''Normalize image by making its standard deviation = 1.0'''
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))
        return img/tf.maximum(std, eps)

def lap_normalize(img, scale_n=4):
    '''Perform the Laplacian pyramid normalization.'''
    img = tf.expand_dims(img,0)
    tlevels = lap_split_n(img, scale_n)
    tlevels = list(map(normalize_std, tlevels))
    out = lap_merge(tlevels)
    return out[0,:,:,:]

def generate_swatches(path="swatches"):
    '''Make swatches for all classes.'''
    oct_n = 4
    oct_s = 1.4
    lap_n = 5
    iter_n = 20
    step = 1.0
    total=0
    for l, layer in tqdm(enumerate(layers)):
        layer = layer.split("/")[1]
        num_channels = T(layer).shape[3]
        num_saved = len([f for f in listdir("%s/%s"%(path,layer)) if isfile(join("%s/%s"%(path,layer), f))])
        total += int(num_channels)
        print(num_channels==num_saved, num_channels, num_saved, layer)
        for n in range(num_saved,num_channels):
            img_noise = np.random.uniform(size=(112,112,3)) + 100.0
            img = render_lapnorm(T(layer)[:,:,:,n], img0=img_noise, iter_n=iter_n, step=step, oct_n=oct_n, oct_s=oct_s, lap_n=lap_n)
            scipy.misc.imsave('%s/%s/c%04d.png'%(path,layer, n), img)
    print("total number of swatches: %d"%total)

def render_lapnorm(t_obj, img0, iter_n=10, step=1.0, oct_n=3, oct_s=1.4, lap_n=4):
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    # build the laplacian normalization graph
    lap_norm_func = tffunc(np.float32)(partial(lap_normalize, scale_n=lap_n))
    img = img0.copy()
    for octave in range(oct_n):
        if octave>0:
            hw = np.float32(img.shape[:2])*oct_s
            img = resize(img, np.int32(hw))
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            g = lap_norm_func(g)
            img += g*step
            print('.', end='')
        print("octave %d/%d"%(octave+1, oct_n))
    #clear_output()
    return img

def lapnorm_multi_old(t_obj, img0, mask, iter_n=10, step=1.0, oct_n=3, oct_s=1.4, lap_n=4):
    t_score = [tf.reduce_mean(t) for t in t_obj] # defining the optimization objective
    t_grad = [tf.gradients(t, t_input)[0] for t in t_score] # behold the power of automatic differentiation!
    # build the laplacian normalization graph
    lap_norm_func = tffunc(np.float32)(partial(lap_normalize, scale_n=lap_n))
    img = img0.copy()
    imgs = []
    for octave in range(oct_n):
        if octave>0:
            hw = np.float32(img.shape[:2])*oct_s
            img = resize(img, np.int32(hw))
            mask = resize(mask, np.int32(hw))
        for i in range(iter_n):
            g_tiled = [lap_norm_func(calc_grad_tiled(img, t)) for t in t_grad]
            for g, gt in enumerate(g_tiled):
                img += gt * step * mask[:,:,g].reshape((mask.shape[0],mask.shape[1],1))
            imgs.append(np.copy(img))
        print("octave %d/%d"%(octave+1, oct_n))
    #clear_output()
    return img, imgs
        
def lapnorm_multi(t_obj, img0, mask, iter_n=10, step=1.0, oct_n=3, oct_s=1.4, lap_n=4, to_clear=True):
    mask_sizes = get_mask_sizes(mask.shape[0:2], oct_n, oct_s)
    t_score = [tf.reduce_mean(t) for t in t_obj] # defining the optimization objective
    t_grad = [tf.gradients(t, t_input)[0] for t in t_score] # behold the power of automatic differentiation!
    # build the laplacian normalization graph
    lap_norm_func = tffunc(np.float32)(partial(lap_normalize, scale_n=lap_n))
    img = img0.copy()
    imgs = []
    for octave in range(oct_n):
        if octave>0:
            hw = mask_sizes[octave] #np.float32(img.shape[:2])*oct_s
            img = resize(img, np.int32(hw))
        oct_mask = resize(mask, np.int32(mask_sizes[octave]))
        for i in range(iter_n):
            g_tiled = [lap_norm_func(calc_grad_tiled(img, t)) for t in t_grad]
            for g, gt in enumerate(g_tiled):
                img += gt * step * oct_mask[:,:,g].reshape((oct_mask.shape[0],oct_mask.shape[1],1))
            imgs.append(np.clip(img, 0, 255))
        print("octave %d/%d"%(octave+1, oct_n))
    if to_clear:
        clear_output()
    return img, imgs

def make_image(mask, channels, img0, iter_n, curr_step, oct_n, oct_s, lap_n, t_clear=True):
    sum_per_channel = mask.sum(axis=0).sum(axis=0)
    active_channels = [i for i in range(len(sum_per_channel)) if sum_per_channel[i]>0]
    active = append_layer_names([channels[a] for a in active_channels])
    mask = mask[:,:,active_channels]
    objectives = [T(a['layer_name'])[:,:,:,a['channel']] for a in active]
    img1, imgs = lapnorm_multi(objectives, img0=img0, mask=mask, iter_n=iter_n, step=curr_step, oct_n=oct_n, oct_s=oct_s, lap_n=lap_n, to_clear=t_clear)
    return img1

def draw_mask_sequence(sequence, attributes, output, flatten_blend=False, draw_rgb=False, animate=False):
    sy, sx = attributes['h'], attributes['w']
    num_frames = sequence.get_num_frames()
    iter_n, step, oct_n, oct_s, lap_n = attributes['iter_n'], attributes['step'], attributes['oct_n'], attributes['oct_s'], attributes['lap_n']
    folder, save_all, save_last = output['folder'], output['save_all'], output['save_last']
    for f in range(num_frames):
        masks, channels, crop, injection = sequence.get_frame(f)
        draw_mask(masks, flatten_blend=flatten_blend, draw_rgb=draw_rgb, animate=animate, color_labels=None)

def generate(sequence, attributes, output, start_from=0, preview=False):
    name, index, save_all, save_last = output['name'], output['index'], output['save_all'], output['save_last']
    sy, sx, iter_n, step, oct_n, oct_s, lap_n, matchHist, brightMult = attributes['h'], attributes['w'], attributes['iter_n'], attributes['step'], attributes['oct_n'], attributes['oct_s'], attributes['lap_n'], attributes['matchHist'], attributes['brightMult']
    mask_sizes = get_mask_sizes([sy, sx], oct_n, oct_s)
    num_frames, num_loop_frames = sequence.get_num_frames(), sequence.get_num_loop_frames()
    
    #name = "%s"%folder
    #name += (output['name'] if 'name' in output else "+".join(["%s-%d"%(c['layer'], c['channel']) for c in sequence.get_channels()]))
    #name += "_n%02d_o%02d_r%0.2f"%(iter_n, oct_n, oct_s)
    
    folder = 'results/%s/%04d'%(name, index)
    make_dir('results/%s'%name)
    make_dir(folder)

    json_data = {'sequences':sequence.get_sequences_json(), 'attributes':attributes, 'output':output}
    with open('%s/config.json'%folder, 'w') as outfile:
        json.dump(json_data, outfile, sort_keys=False, indent=4)
    
    hist = None
    if start_from == 0:
        img = np.random.uniform(size=(sy, sx, 3)) + 100.0
    else:
        img = scipy.misc.imread('%s/f%04d.png'%(folder, 1+(start_from-1)%num_frames))
        hist = get_histogram(img.astype('uint8'), bright=False)

    for f in range(start_from, num_frames + num_loop_frames):
        mask, channels, canvas, injection = sequence.get_frame(f)
        img = match_histogram(img, hist) if matchHist and hist is not None else img
        img = 128.0 + 0.75 * (img - 128.0) if brightMult < 1.0 else img

        # brightMult?

        img = modify_canvas(img, canvas)
        img = resize(img, np.int32(mask_sizes[0])) 

        if injection != None:
            img = inject_image(img, injection['path'], injection['amt'])

        if f >= num_frames:
            max_amt = 0.4  # hmmmm...
            amt = max_amt * (f-num_frames+1)/num_loop_frames
            img_in = scipy.misc.imread('%s/f%04d.png'%(folder, 1+f%num_frames))
            img_in = resize(img_in, np.int32(mask_sizes[0]))
            img = (1.0 - amt) * img + amt * img_in

        img = make_image(mask, channels, img, iter_n, step, oct_n, oct_s, lap_n, save_all)
        img = np.uint8(np.clip(img, 0, 255))
        hist = get_histogram(np.uint8(img), bright=False)

        if save_all or f == num_frames-1:
            scipy.misc.imsave('%s/f%04d.png'%(folder, 1+f%num_frames), img)

            output['last_frame'] = f
            json_data = {'sequences':sequence.get_sequences_json(), 'attributes':attributes, 'output':output}
            with open('%s/config.json'%folder, 'w') as outfile:
                json.dump(json_data, outfile, sort_keys=False, indent=4)

        # if save_last and f == num_frames-1:
        #     scipy.misc.imsave('%s/%04d.png'%(folder, idx), img)
        if preview:
            showarray(img/255.)
    
    return img




def merge(config1, config2, out_name, out_index, start_from, margin):
    sequence1, attributes1, output1 = load_config('%s/config.json'%config1)
    sequence2, attributes2, output2 = load_config('%s/config.json'%config2)
    #generate(sequence, attributes, output, start_from=start_from, preview=False)
    
    sequences1 = sequence1.get_sequences_json()
    sequences2 = sequence2.get_sequences_json()
    

    #start_from = sequences1[-1]['time'][1]
    n1, n2 = start_from, sequences2[-1]['fade_out'][1]
    

    # start_from may not actually be in the last seq
    sequences1[-1]['time'][1] = start_from
    sequences1[-1]['fade_out'] = [start_from, start_from + margin]
    

    f = start_from
    
    nf = sequences2[0]['time'][1]-sequences2[0]['time'][0]
    
    nfi = sequences2[0]['fade_in'][1]-sequences2[0]['fade_in'][0]
    nfo = sequences2[0]['fade_out'][1]-sequences2[0]['fade_out'][0]


    # sequences2[0]['fade_in'] = [f, f + margin]
    # sequences2[0]['time'] = [f + margin, f + margin + nf]
    # sequences2[0]['fade_out'] = [f + margin + nf, f + margin + nf + nfo]

    sequences2[0]['fade_in'] = [f, f + margin]
    sequences2[0]['time'] = [f + margin, f + nf]
    sequences2[0]['fade_out'] = [f + nf, f + nf + nfo]


    f = f + nf
    for s in range(1, len(sequences2)):
        nf = sequences2[s]['time'][1]-sequences2[s]['time'][0]
        nfi = sequences2[s]['fade_in'][1]-sequences2[s]['fade_in'][0]
        nfo = sequences2[s]['fade_out'][1]-sequences2[s]['fade_out'][0]

        # sequences2[s]['fade_in'] = [f, f + nfi]
        # sequences2[s]['time'] = [f + nfi, f + nfi + nf]
        # sequences2[s]['fade_out'] = [f + nfi + nf, f + nfi + nf + nfo]

        # f = f + nfi + nf + nfo

        sequences2[s]['fade_in'] = [f, f + nfi]
        sequences2[s]['time'] = [f + nfi, f + nfi + nf]
        sequences2[s]['fade_out'] = [f + nfi + nf, f + nfi + nf + nfo]

        print("SEQ",s,f,f+nfi,f+nfi+nf,f+nfi+nf+nfo)

        f = f + nfi + nf    
        print("f now",f)

        # sequences2[0]['fade_in'] = [start_from, start_from + margin]
        # sequences2[0]['time'] = [start_from + margin, start_from + n2]
        # sequences2[0]['fade_out'][0] += start_from
        # sequences2[0]['fade_out'][1] += start_from
    

    # have to adjust times for each subsequent s in sequences2
    
    sequence3 = Sequence(attributes1)
    for s in sequences1:
        sequence3.append_json(s)
    for s in sequences2:
        sequence3.append_json(s)
    
    
    
    
    
    name1, index1 = output1['name'], output1['index']
    name2, index2 = output2['name'], output2['index']

    name3, index3 = out_name, out_index
    output3 = output1
    output3['name'] = out_name
    output3['index'] = out_index
    if 'last_frame' in output3:
        del output3['last_frame']
        
    
    folder1 = 'results/%s/%04d'%(name1, index1)
    folder2 = 'results/%s/%04d'%(name2, index2)
    folder3 = 'results/%s/%04d'%(name3, index3)
    make_dir('results/%s'%name3)
    make_dir(folder3)


    json_data = {'sequences':sequence3.get_sequences_json(), 'attributes':attributes1, 'output':output3}
    with open('%s/config.json'%folder3, 'w') as outfile:
        json.dump(json_data, outfile, sort_keys=False, indent=4)

    


    for f in range(1, start_from+1):
        cmd = 'cp %s/f%04d.png %s/f%04d.png'%(folder1, f, folder3, f)
        print(cmd)
        os.system(cmd)
        
    last_frame = n2
    for f in range(1, last_frame):
        f2 = start_from + f
        cmd = 'cp %s/f%04d.png %s/f%04d.png'%(folder2, f, folder3, f2)
        print(cmd)
        os.system(cmd)

    # make dir
    # copy files to folder
    
    
    # add mask2 to sequence (so they interpolate)
    


    
    sy, sx, iter_n, step, oct_n, oct_s, lap_n, matchHist, brightMult = attributes1['h'], attributes1['w'], attributes1['iter_n'], attributes1['step'], attributes1['oct_n'], attributes1['oct_s'], attributes1['lap_n'], attributes1['matchHist'], attributes1['brightMult']
    mask_sizes = get_mask_sizes([sy, sx], oct_n, oct_s)
    num_frames = 100000 # doesn't matter right now

    
    img = scipy.misc.imread('%s/f%04d.png'%(folder3, 1+(start_from-1)%num_frames))
    print("LOAD!",img.shape)
    hist = get_histogram(img.astype('uint8'), bright=False)

    for f in range(start_from, start_from + margin):
        mask, channels, canvas, injection = sequence3.get_frame(f)
        img = match_histogram(img, hist) if matchHist and hist is not None else img
        img = 128.0 + 0.75 * (img - 128.0) if brightMult < 1.0 else img

        # brightMult?

        img = modify_canvas(img, canvas)
        img = resize(img, np.int32(mask_sizes[0])) 

#         if injection != None:
#             img = inject_image(img, injection['path'], injection['amt'])
        f2 = 1 + f - start_from
        max_amt = 1.0  # hmmmm...
        amt = max_amt * f2 / margin
        print("MAKE! ",f, f2, amt, "input old", 1+f%num_frames)
        img_in = scipy.misc.imread('%s/f%04d.png'%(folder3, 1+f%num_frames))
        img_in = resize(img_in, np.int32(mask_sizes[0]))
        img = (1.0 - amt) * img + amt * img_in

        img = make_image(mask, channels, img, iter_n, step, oct_n, oct_s, lap_n, True)
        img = np.uint8(np.clip(img, 0, 255))
        hist = get_histogram(np.uint8(img), bright=False)
        #print('make -> %s/f%04d.png'%(folder3, 1+f%num_frames))
        

        scipy.misc.imsave('%s/f%04d.png'%(folder3, 1+f%num_frames), img)

            # output['last_frame'] = f
            # json_data = {'sequences':sequence.get_sequences_json(), 'attributes':attributes, 'output':output}
            # with open('%s/config.json'%folder, 'w') as outfile:
            #     json.dump(json_data, outfile, sort_keys=False, indent=4)

