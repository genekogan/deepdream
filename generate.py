## %matplotlib inline
import IPython.display
import PIL
from PIL import Image, ImageStat, ImageEnhance
from io import BytesIO
import math
import sys, argparse
import random
import os
from os import listdir
from os.path import isfile, isdir, join
import numpy as np
import scipy.misc
from tqdm import tqdm

ImageEnhance.LOAD_TRUNCATED_IMAGES = True
ImageStat.LOAD_TRUNCATED_IMAGES = True
Image.LOAD_TRUNCATED_IMAGES = True

def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 1)*255)
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))
    
def visstd(a, s=0.1):
    '''Normalize the image range for visualization'''
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

def histogram_equalization(x):
    hist, bins = np.histogram(x.flatten(), 255, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255.0/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    x2 = cdf[x.astype('uint8')]
    return x2, cdf

def get_histogram(pixels, bright=True):
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

def match_histogram2(img1, hist):
    colors = img1.getdata()
    red, green, blue = [c[0] for c in colors], [c[1] for c in colors], [c[2] for c in colors]
    sr = sorted(range(len(red)), key=lambda k: red[k])
    sg = sorted(range(len(green)), key=lambda k: green[k])
    sb = sorted(range(len(blue)), key=lambda k: blue[k])
    hr, hg, hb = [[hist[i][c] for i in range(256)] for c in range(3)]
    fr, fg, fb = 0,0,0
    for c in range(len(hr)):
        nfr, nfg, nfb = int(hr[c]), int(hg[c]), int(hb[c])
        idxr = [sr[k] for k in range(fr,fr+nfr)]
        idxg = [sg[k] for k in range(fg,fg+nfg)]
        idxb = [sb[k] for k in range(fb,fb+nfb)]
        for ir in idxr:
            red[ir] = c
        for ig in idxg:
            green[ig] = c
        for ib in idxb:
            blue[ib] = c
        fr, fg, fb = fr+nfr, fg+nfg, fb+nfb
    adjusted_colors = zip(red, green, blue)
    img_adjusted = Image.new(img1.mode, img1.size)
    img_adjusted.putdata(adjusted_colors)
    return img_adjusted

def match_histogram(img1, hist):
    pixels = list(img1.getdata())
    red, green, blue = np.array([c[0] for c in pixels]), np.array([c[1] for c in pixels]), np.array([c[2] for c in pixels])
    sr = sorted(range(len(red)), key=lambda k: red[k])
    sg = sorted(range(len(green)), key=lambda k: green[k])
    sb = sorted(range(len(blue)), key=lambda k: blue[k])
    num_pixel_mult = (3 * len(pixels)) / np.sum(hist)
    hr, hg, hb = [[int(num_pixel_mult * hist[i][c]) for i in range(256)] for c in range(3)]
    fr, fg, fb = 0, 0, 0
    for c in range(len(hr)):
        nfr, nfg, nfb = int(hr[c]), int(hg[c]), int(hb[c])
        red[np.array([sr[k] for k in xrange(fr,fr+nfr)]).astype('int')] = c
        green[np.array([sg[k] for k in xrange(fg,fg+nfg)]).astype('int')] = c
        blue[np.array([sb[k] for k in xrange(fb,fb+nfb)]).astype('int')] = c
        fr, fg, fb = fr+nfr, fg+nfg, fb+nfb
    adjusted_pixels = zip(red, green, blue)
    img_adjusted = Image.new(img1.mode, img1.size)
    img_adjusted.putdata(adjusted_pixels)
    return img_adjusted

def adjust_color_range(img, hist, amt, border):
    cdf = hist.cumsum() / np.sum(hist)
    i1, i2 = min([i for i in range(256) if cdf[i]>border]), max([i for i in range(256) if cdf[i]<1.0-border])
    j1, j2 = int((1.0-amt)*i1), i2 + amt*(255-i2)
    img2 = np.clip(j1 + (j2-j1)*(img - i1)/(i2-i1), 0.0, 255.0)
    return img2

def get_average_histogram(frames_path):
    numframes = len([f for f in listdir(frames_path) if isfile(join(frames_path, f)) and f[-4:]=='.png'])
    img = Image.open('%s/f0001.png'%(frames_path))
    histogram = get_histogram(list(img.getdata()))
    for t in tqdm(range(1,numframes,8)):
        img = Image.open('%s/f%04d.png'%(frames_path, t+1))
        histogram += get_histogram(list(img.getdata()))
    histogram /= (1+len(range(1,numframes,8)))
    return histogram

def generate_video2(frames_path, output_path):
    numframes = len([f for f in listdir(frames_path) if isfile(join(frames_path, f))])
    print('loading %d images from %s' % (numframes, frames_path))
    images = [ Image.open('%s/f%04d.png'%(frames_path, t+1)) for t in range(numframes) ]
    brightness = [ ImageStat.Stat(img.convert('L')).mean[0] for img in images ]
    avg_brightness = np.mean(brightness)
    os.system('mkdir %s/temp' % frames_path)
    for i, img in tqdm(enumerate(images)):
        mult = avg_brightness / brightness[i]
        source = img.split()
        r = source[0].point(lambda i: i*mult)
        g = source[1].point(lambda i: i*mult)
        b = source[2].point(lambda i: i*mult)
        img2 = Image.merge(img.mode, (r, g, b))
        img2 = ImageEnhance.Contrast(img2).enhance(1.2)
        img2 = ImageEnhance.Sharpness(img2).enhance(1.25)
        scipy.misc.imsave("%s/temp/f%04d.png" % (frames_path, i+1), img2)
    w, h  = images[0].size
    cmd = 'ffmpeg -i %s/temp/f%%04d.png -c:v libx264 -pix_fmt yuv420p -vf scale=%d:%d %s'%(frames_path, w-(w%2), h-(h%2), output_path)
    print('creating movie at %s' % output_path)
    os.system('rm %s' % output_path)
    os.system(cmd)
    os.system('rm -rf %s/temp' % frames_path)
    print('done!')

def generate_video(frames_path, output_path, sat, con, sharp, bitrate=None, cumulative=False):
    ImageEnhance.LOAD_TRUNCATED_IMAGES = True
    ImageStat.LOAD_TRUNCATED_IMAGES = True
    Image.LOAD_TRUNCATED_IMAGES = True

    numframes = len([f for f in listdir(frames_path) if isfile(join(frames_path, f)) and f[-4:]=='.png'])
    if numframes == 0:
        print("no frames found in %s"%frames_path)
        return
    print('creating %d-frame movie: %s -> %s'%(numframes, frames_path, output_path))
#     avg_hist = get_average_histogram(frames_path)
    os.system('mkdir %s/temp' % frames_path)
    for i in tqdm(range(numframes)):
        edited_filepath = "%s/temp/f%04d.png" % (frames_path, i+1)
        if not os.path.isfile(edited_filepath) or not cumulative:
            img = Image.open('%s/f%04d.png'%(frames_path, i+1))
    #         img = match_histogram(img, avg_hist)
            img = ImageEnhance.Color(img).enhance(sat)
            img = ImageEnhance.Contrast(img).enhance(con)
            img = ImageEnhance.Sharpness(img).enhance(sharp)
            scipy.misc.imsave(edited_filepath, img)
    w, h  = img.size
    wx, wy = w-(w%2), h-(h%2)
    #wx, wy = 600, 360
    if bitrate==None:
        cmd = 'ffmpeg -i %s/temp/f%%04d.png -c:v libx264 -pix_fmt yuv420p -vf scale=%d:%d %s'%(frames_path, wx, wy, output_path)
    else:
        cmd = 'ffmpeg -i %s/temp/f%%04d.png -c:v libx264 -pix_fmt yuv420p -vf scale=%d:%d -b %d %s'%(frames_path, wx, wy, bitrate, output_path)
    #cmd = 'ffmpeg -i %s/temp/frame%%04d.png -c:v libx264 -preset ultrafast -crf 0 -vf scale=%d:%d %s'%(frames_path, w-(w%2), h-(h%2), output_path)
    os.system('rm %s' % output_path)
    os.system(cmd)
    if not cumulative:
        os.system('rm -rf %s/temp' % frames_path)
    print('done!')


def gen_video(frames_path, sat, con, sharp, bitrate=None, cumulative=False):
    output_path = '%s_%0.2f,%0.2f,%0.2f.mp4'%(frames_path, sat, con, sharp)
    generate_video(frames_path, output_path, sat, con, sharp, bitrate, cumulative)

def gen_video_dir(frames_path, sat, con, sharp, overwrite=True, bitrate=None, cumulative=False):
    dirs = [f for f in listdir(frames_path) if isdir(join(frames_path, f))]
    for d in dirs:
        output_path = '%s/%s_%0.2f,%0.2f,%0.2f.mp4'%(frames_path, d, sat, con, sharp)
        file_exists = os.path.isfile(output_path)
        if overwrite or not file_exists:
            print("generate %s"%output_path)
            gen_video(join(frames_path, d), sat, con, sharp, bitrate, cumulative)


def process_arguments(args):
    parser = argparse.ArgumentParser(description='generate video')
    parser.add_argument('--video', action='store', help='path to directory of input video')
    parser.add_argument('--dir', action='store', help='path to directory of directories of input videos')
    parser.add_argument('--sacosh', action='store', help='saturation, contrast, sharpness')
    params = vars(parser.parse_args(args))
    return params

if __name__ == '__main__':
    params = process_arguments(sys.argv[1:])
    sa, co, sh = 1.0, 1.0, 1.0
    if params['sacosh'] is not None:
        sacosh = params['sacosh'].split(',')
        sa, co, sh = float(sacosh[0]), float(sacosh[1]), float(sacosh[2])
    if params['video'] is not None:
        gen_video(params['video'], sa, co, sh)
    elif params['dir'] is not None:
        gen_video_dir(params['dir'], sa, co, sh, overwrite=True)

#s,c,sh = 1.1,1.25,1.2

#gen_video_dir("/home/gene/lapnorm/results/test_5",s, c, sh, overwrite=True)
#gen_video("/home/gene/lapnorm/results/test_7/F2-41+C2-0+C3-179+B2-44+C1-106_n11_o02_r1.39",s, c, sh)
#gen_video("/home/gene/lapnorm/results/test_8/F2-41+C2-0+C3-179+B2-44+C1-106_n11_o02_r1.39",s, c, sh)

#gen_video("/home/gene/lapnorm/results/new2/C1-1_n10_o03_r1.32", s, c, sh, None, False)

#gen_video("/home/gene/lapnorm/results/new2/D1-152+D1-120+H4-13+G2-30+H3-103_n15_o02_r1.32", s, c, sh, None, False)
#gen_video("/home/gene/lapnorm/results/new3/C6-54+D2-50+C2-6+C6-5_n12_o02_r1.32", s, c, sh, None, False)



#gen_video("/home/gene/lapnorm/results/new3/C1-7+C6-30+B3-62+C6-5_n10_o02_r1.32", s, c, sh, None, False)
#gen_video("/home/gene/lapnorm/results/new2/F5-50+D3-100+F6-54+D1-162_n15_o03_r1.32", s, c, sh, None, False)

# faves = ['/home/gene/lapnorm/results/new2/C1-1_n10_o03_r1.32', 
#         '/home/gene/lapnorm/results/new2/H6-88_n10_o03_r1.32',
#         '/home/gene/lapnorm/results/new2/C6-5_n10_o03_r1.32',
#         '/home/gene/lapnorm/results/new2/H3-77_n10_o03_r1.32',
#         '/home/gene/lapnorm/results/new2/G2-36_n10_o03_r1.32',
#         '/home/gene/lapnorm/results/new2/H1-183_n10_o03_r1.32']
# for g in faves:
#     gen_video(g, s, c, sh, None, False)
