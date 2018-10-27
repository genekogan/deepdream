from __future__ import print_function
import math
import time
import copy
import json
import sys, argparse
import os
import os.path
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
from lapnorm import *
from canvas import *
from mask import *
from util import *
from bookmarks import *
from generate import *


# todo:
#   keep track of last_frame generated
#   make naming + numbering unecessary in the json -> gets made auto in deploy and into config.json
#   temporarily put "in-progress" tag? (but what if it crashes)


def remove_empty_queue_folders(queue_dir):
	all_config_folders = glob.glob(os.path.join(queue_dir, '*'))
	for config_folder in all_config_folders:
		if len(glob.glob(os.path.join(config_folder, '*'))) == 0:
			os.system('rm -rf %s'%config_folder)


def deploy_loop(queue_dir):
	while True:

		all_configs = glob.glob(os.path.join(queue_dir, '**/*.json'))
		all_configs = sorted(all_configs, key=os.path.getctime)
		active_config = None
		while active_config is None and len(all_configs)>0:
			active_config = all_configs.pop(0)
			print("do ",active_config)
			sequence, attributes, output = load_config(active_config)
			if 'busy' in output and output['busy']==True:
				print("busy: %s"%active_config)
				active_config = None

		if active_config is None:
			print("none left, exit")
			remove_empty_queue_folders(queue_dir)
			return

		else:
			print("begin to generate: %s"%active_config)
			output['busy'] = True
			json_data = {'sequences':sequence.get_sequences_json(), 'attributes':attributes, 'output':output}
			with open(active_config, 'w') as outfile:
				json.dump(json_data, outfile, sort_keys=False, indent=4)

			start_from = 0
			folder = 'results/%s/%04d'%(output['name'], output['index'])
			if os.path.isfile('%s/config.json' % folder): 
				with open('%s/config.json' % folder, 'r') as infile:
					json_data = json.load(infile)
					if 'last_frame' in json_data['output']:
						print("found sf",start_from)
						start_from = json_data['output']['last_frame']

			print("start from",start_from)

			generate(sequence, attributes, output, start_from=start_from, preview=False)
			os.system('rm %s'%active_config)
			print("finished generating: %s"%active_config)
			remove_empty_queue_folders(queue_dir)



def process_arguments(args):
    parser = argparse.ArgumentParser(description='deploy')
    parser.add_argument('--unbusy', action='store', help='path to directory to unbusy')
    parser.add_argument('--config1', action='store', help='path to directory of config1')
    parser.add_argument('--config2', action='store', help='path to directory of config2')
    parser.add_argument('--out_name', action='store', help='path to directory of output')
    parser.add_argument('--out_index', action='store', help='index of output name')
    parser.add_argument('--start_from', action='store', help='merge start frame')
    parser.add_argument('--margin', action='store', help='merge margin')
    params = vars(parser.parse_args(args))
    return params


if __name__ == '__main__':
	params = process_arguments(sys.argv[1:])
	sa, co, sh = 1.0, 1.0, 1.0
	if params['unbusy'] is not None:
		all_configs = glob.glob(os.path.join('results/_queue/%s'%params['unbusy'], '*.json'))
		for config in all_configs:
			sequence, attributes, output = load_config(config)
			if 'busy' in output and output['busy']==True:
				output['busy']=False
				json_data = {'sequences':sequence.get_sequences_json(), 'attributes':attributes, 'output':output}
				with open(config, 'w') as outfile:
					json.dump(json_data, outfile, sort_keys=False, indent=4)
	
	elif params['config1'] is not None and params['config2'] is not None:
		config1, config2, out_name, out_index, start_from, margin = params['config1'], params['config2'], params['out_name'], int(params['out_index']), int(params['start_from']), int(params['margin'])
		merge(config1, config2, out_name, out_index, start_from, margin)

	else:
		deploy_loop('results/_queue/')


#python deploy.py --config1 results/arcs4/0005 --config2 results/arcs4/0001 --out_name nips_final --out_index 1 --start_from 300 --margin 130