import numpy as np
import json
from util import *

dir1 ='/home/gene/lapnorm/results/interps'
dir2 = '/home/gene/lapnorm/results/interps2'

js=[45, 42, 40, 39, 38, 37, 35, 31, 26, 23, 22, 19, 18, 17, 13, 9, 6, 5, 2, 1]

for j in js:
	f1 = '%s/%04d/config.json'%(dir1,j)
	f2 = '%s/%04d.json'%(dir2,j)
	sequence, attributes, output = load_config(f1)
	output['index'] = j
	output['name'] = 'interps2'
	attributes['w'] = 1920
	attributes['h'] = 1080
	attributes['oct_n'] = 3
	attributes['oct_s'] = 1.7197235
	if 'last_frame' in output:
		del output['last_frame']
	sequences = sequence.get_sequences_json()
	sequences[0]['time'] = [0,500]
	sequences[0]['fade_out'] = [0,500]
	sequences[0]['mask']['period'] = 100
	json_data = {'sequences':sequences, 'attributes':attributes, 'output':output}
	with open(f2, 'w') as outfile:
		json.dump(json_data, outfile, sort_keys=False, indent=4)

	