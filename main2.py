from __future__ import print_function
import math
import time
import copy
import json
import os
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

# reloading external code so notebook updates properly
from importlib import reload
import lapnorm, mask, util, bookmarks
reload(lapnorm)
reload(mask)
reload(util)
reload(bookmarks)
from lapnorm import *
from mask import *
from util import *
from bookmarks import *


def make_one():

    nf = 3

    lo_res = 2*random()
    vert = 3*random()
    znc = int(random()*3)
    zc = int(random()*3)
    zm = int(random()*6)

    if lo_res < 1.0:
        d1, d2 = 1080, 1920
    else:
        d1, d2 = 3508, 4968

    if vert < 1.0:
        h, w = d2, d1
    else:
        h, w = d1, d2
    
    output = {'folder':'results/grid9/', 'save_all':False, 'save_last':True}
    attributes = {'h':h, 'w':w, 'iter_n':36, 'step':1.0, 'oct_n':6, 'oct_s':1.739, 'lap_n':24}

    nc = 2 if znc==0 else int(2+3*random())
    blend_ = 0.6*random() if znc==0 else 0.3*random()
    blend_ += 0.1

    if zc == 0:
        channels = get_random_favorites(faves_m5, nc)
    elif zc == 1:
        channels = get_random_favorites(faves_m4, nc)
    elif zc == 2:
        channels = get_random_favorites(layers_m3, nc)

    if zm == 0:
        mask = {'type':'arcs', 'h':h, 'w':w, 'n':nc, 'period':nf, 'ctr_y':0.5, 'ctr_x':0.5, 'radius':0.70710678118/nc, 'blend':blend_, 'inwards':True, 'reverse':True, 'crop':Crop.NONE}
    elif zm == 1:
        mask = {'type':'arcs', 'h':h, 'w':w, 'n':nc, 'period':nf, 'ctr_y':1.0, 'ctr_x':0.5, 'radius':1.1181/nc, 'blend':blend_, 'inwards':True, 'reverse':True, 'crop':Crop.NONE}
    elif zm == 2:
        mask = {'type':'arcs', 'h':h, 'w':w, 'n':nc, 'period':nf, 'ctr_y':0.0, 'ctr_x':0.0, 'radius':1.415/nc, 'blend':blend_, 'inwards':True, 'reverse':True, 'crop':Crop.NONE}
    elif zm == 3:
        mask = {'type':'rects', 'h':h, 'w':w, 'n':nc, 'period':nf, 'p1':(0.0,0.0), 'p2':(1.0,1.0), 'width':2.0, 'blend':blend_, 'reverse':True, 'crop':Crop.NONE}
    elif zm == 4:
        mask = {'type':'rects', 'h':h, 'w':w, 'n':nc, 'period':nf, 'p1':(0.5,0.0), 'p2':(0.5,1.0), 'width':2.0, 'blend':blend_, 'reverse':True, 'crop':Crop.NONE}
    elif zm == 5:
        mask = {'type':'rects', 'h':h, 'w':w, 'n':nc, 'period':nf, 'p1':(0.0,0.5), 'p2':(1.0,0.5), 'width':2.0, 'blend':blend_, 'reverse':True, 'crop':Crop.NONE}
    print(output)
    print(attributes)
    print(channels)
    print(mask)
    
    sequence = Sequence(attributes['h'], attributes['w'], attributes['oct_n'], attributes['oct_s'])
    sequence.append(channels, mask, nf, 0)

    img=generate(sequence, attributes, output, start_from=0, preview=False)

    

for z in range(1000):
    make_one()



