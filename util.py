from __future__ import print_function
import os
from os import listdir
from os.path import isdir, isfile, join
import numpy as np
import json
from bookmarks import *
from io import BytesIO
import PIL.Image
from PIL import ImageDraw, ImageFont
#from IPython.display import clear_output, Image, display, HTML
from mask import *
from canvas import *


def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 1)*255)
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

def histogram_equalization(x):
    hist, bins = np.histogram(x.flatten(), 255, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255.0/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    x2 = cdf[x.astype('uint8')]
    return x2, cdf

def get_histogram(img, bright=True):
    w, h = img.shape[0:2]
    pixels = img.reshape(w*h, 3)
    hist = np.zeros((256,1 if bright else 3))
    for p in pixels:
        if bright:
            avg = int((p[0]+p[1]+p[2])/3.0)
            hist[avg,0] += 1
        else:
            hist[p[0],0] += 1
            hist[p[1],1] += 1
            hist[p[2],2] += 1
    return np.array(hist)

def match_histogram(img, hist):
    w, h = img.shape[0:2]
    pixels = img.reshape(w*h, 3)
    red, green, blue = np.array([c[0] for c in pixels]), np.array([c[1] for c in pixels]), np.array([c[2] for c in pixels])
    sr = sorted(range(len(red)), key=lambda k: red[k])
    sg = sorted(range(len(green)), key=lambda k: green[k])
    sb = sorted(range(len(blue)), key=lambda k: blue[k])
    num_pixel_mult = (3 * len(pixels)) / np.sum(hist)
    hr, hg, hb = [[int(num_pixel_mult * hist[i][c]) for i in range(256)] for c in range(3)]
    fr, fg, fb = 0, 0, 0
    for c in range(len(hr)):
        nfr, nfg, nfb = int(hr[c]), int(hg[c]), int(hb[c])
        red[np.array([sr[k] for k in range(fr,fr+nfr)]).astype('int')] = c
        green[np.array([sg[k] for k in range(fg,fg+nfg)]).astype('int')] = c
        blue[np.array([sb[k] for k in range(fb,fb+nfb)]).astype('int')] = c
        fr, fg, fb = fr+nfr, fg+nfg, fb+nfb
    adjusted_pixels = np.array(list(zip(red, green, blue)))
    adjusted_image = adjusted_pixels.reshape(w, h, 3)
    return adjusted_image

def adjust_color_range(img, hist, amt, border):
    cdf = hist.cumsum() / np.sum(hist)
    i1, i2 = min([i for i in range(256) if cdf[i]>border]), max([i for i in range(256) if cdf[i]<1.0-border])
    j1, j2 = int((1.0-amt)*i1), i2 + amt*(255-i2)
    img2 = np.clip(j1 + (j2-j1)*(img - i1)/(i2-i1), 0.0, 255.0)
    return img2

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def append_layer_names(channels):
    for c in channels:
        c['layer_name'] = layer_lookup[c['layer']]
    return channels

def get_channels_from_filename(name):
    channels = []
    name = name.split('_')[0].split('+')
    for n in name:
        channels.append({'channel': int(n.split('-')[1]), 'layer': n.split('-')[0]})
    return channels

def load_config(path):
    with open('%s' % path, 'r') as infile:
        json_data = json.load(infile)
    sequences, attributes, output = json_data['sequences'], json_data['attributes'], json_data['output']
    sequence = Sequence(attributes)
    sequence.loop_to_beginning(output['num_loop_frames'])
    for seq in sequences:
        sequence.append_json(seq)
    return sequence, attributes, output

def push_to_queue(sequence, attributes, output):
    # queue_file = 'results/_queue/_queue.json'
    name = output['name']
    make_dir('results/_queue/%s'%name)
    subdirs = glob.glob(os.path.join('results/_queue/%s'%name, '*.json'))
    idx_gen = int(max(subdirs, key=os.path.getmtime).split('.json')[-2].split('/')[-1]) + 1 if len(subdirs) > 0 else 1
    output['index'] = idx_gen
    config_file = 'results/_queue/%s/%04d.json'%(name, idx_gen)
    json_data = {'sequences':sequence.get_sequences_json(), 'attributes':attributes, 'output':output}
    with open(config_file, 'w') as outfile:
        json.dump(json_data, outfile, sort_keys=False, indent=4)
    
def make_summary_image_dir(path):
    font = ImageFont.truetype("/home/gene/lapnorm/media/Arial.ttf", 20)
    summary_path = '%s/summary'%path
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    dirs = [f for f in listdir(path) if isdir(join(path, f)) and f != 'summary']
    for d in dirs:
        print(d)
        files = [f for f in listdir(join(path, d)) if isfile(join(join(path, d), f)) and f[-4:]=='.png']    
        if len(files) == 0:
            continue
        json_file = join(join(path, d), 'config.json')
        path_last = join(join(path, d), files[-1])
        with open('%s' % json_file, 'r') as infile:
            json_data = json.load(infile)
        sequences, attributes, output = json_data['sequences'], json_data['attributes'], json_data['output']
        channel_str_file = '+'.join(['%s-%d'%(c['layer'],c['channel']) for c in sequences[0]['channels']])
        channel_str_draw = '   '.join(['%s-%d'%(c['layer'],c['channel']) for c in sequences[0]['channels']])
        path_out = join(summary_path, '%s_%s.png'%(d, channel_str_file))
        img = PIL.Image.open(path_last, "r")
        draw = ImageDraw.Draw(img, 'RGBA')
        draw.rectangle((8, 8, 444, 36), fill=(0, 0, 0, 175), outline=None)
        draw.text((10,10), channel_str_draw, (255, 255, 255, 255),font=font)
        img.save(path_out)


#make_summary_image_dir('/home/gene/lapnorm/results/nipsfaces2')