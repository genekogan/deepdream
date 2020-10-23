from tqdm import tqdm
import numpy as np
import tensorflow as tf
from util import *


__deepdream_config_keys__ = [
    'size', 'objective', 'num_octaves', 
    'octave_ratio', 'num_iterations', 'masks',
    'lap_n', 'step', 'normalize_gradients', 
    'grayscale_gradients'
]


def run_deepdream(deepdream, config, img=None, title=None):
    cfg = EasyDict(config)
    
    cfg.objective = cfg.objective if isinstance(cfg.objective, list) else [cfg.objective]
    cfg.size = cfg.size if 'size' in cfg else 512
    cfg.masks = cfg.masks if 'masks' in cfg else None 
    cfg.num_octaves = cfg.num_octaves if 'num_octaves' in cfg else 1
    cfg.octave_ratio = cfg.octave_ratio if 'octave_ratio' in cfg else 1.0
    cfg.num_iterations = cfg.num_iterations if 'num_iterations' in cfg else 10
    cfg.num_iterations = cfg.num_iterations if isinstance(cfg.num_iterations, list) else [cfg.num_iterations]
    cfg.num_iterations = cfg.num_iterations * cfg.num_octaves if len(cfg.num_iterations) == 1 else cfg.num_iterations
    cfg.step = cfg.step if 'step' in cfg else 1.0
    cfg.step = cfg.step if isinstance(cfg.step, list) else [cfg.step]
    cfg.step = cfg.step * cfg.num_octaves if len(cfg.step) == 1 else cfg.step
    cfg.lap_n = cfg.lap_n if 'lap_n' in cfg else 8
    cfg.lap_n = cfg.lap_n if isinstance(cfg.lap_n, list) else [cfg.lap_n]
    cfg.lap_n = cfg.lap_n * cfg.num_octaves if len(cfg.lap_n) == 1 else cfg.lap_n
    cfg.normalize_gradients = cfg.normalize_gradients if 'normalize_gradients' in cfg else False
    cfg.grayscale_gradients = cfg.grayscale_gradients if 'grayscale_gradients' in cfg else False
    
    extraneous_keys = [k for k in cfg.keys() if k not in __deepdream_config_keys__]
    assert len(extraneous_keys) == 0, \
        'Following config keys are not recognized: %s' % ', '.join(extraneous_keys)
    assert cfg.num_octaves == len(cfg.num_iterations), \
        'Error: must have cfg.num_octaves elements in cfg.num_iterations list'
    assert cfg.num_octaves == len(cfg.step), \
        'Error: must have cfg.num_octaves elements in cfg.step list'
    assert cfg.num_octaves == len(cfg.lap_n), \
        'Error: must have cfg.num_octaves elements in cfg.lap_n list'
        
    # load input image
    if img is None:
        if not isinstance(cfg.size, tuple):
            cfg.size = (cfg.size, cfg.size)
        img = np.random.uniform(size=(cfg.size[1], cfg.size[0], 3)) + 100.0
    else:
        if not isinstance(cfg.size, tuple):
            cfg.size = (int(get_aspect_ratio(img) * cfg.size), cfg.size)
        if isinstance(img, str):
            img = load_image(img, cfg.size)
        else:
            img = resize(img.copy(), cfg.size)
    img = np.array(img).astype(np.float32)
    
    # load masks
    if cfg.masks is not None:    
        if isinstance(cfg.masks, np.ndarray):
            masks_orig = cfg.masks
            masks_orig = masks_orig if masks_orig.ndim==3 else np.expand_dims(masks_orig, 2)
        else:
            cfg.masks = cfg.masks if isinstance(cfg.masks, list) else [cfg.masks]
            mask_images = [load_image(mask, cfg.size, to_numpy=True, normalize=True) 
                           for mask in cfg.masks]
            mask_images = [np.mean(mask_image, -1) for mask_image in mask_images]
            masks_orig = np.transpose(np.array(mask_images), (1, 2, 0))

        assert masks_orig.shape[-1] == len(cfg.objective), \
            "Error: number of masks doesn't match number of objectives"
    
    # save upsampling distortion error for octave correction later 
    octaves = []
    for octave in range(cfg.num_octaves-1):
        hw = img.shape[:2]
        octave_size = np.int32(np.float32(hw)/cfg.octave_ratio)
        lo = deepdream.resize(img, octave_size)
        hi = img-deepdream.resize(lo, hw)
        img = lo
        octaves.append(hi)

    # setup tensorflow gradients
    t_objectives = [deepdream.T(obj['layer'])[:,:,:,obj['channel']] 
                    for obj in cfg.objective]
    t_scores = [tf.reduce_mean(t_obj) 
                for t_obj in t_objectives]
    t_grads = [tf.gradients(t_score, deepdream.t_input)[0] 
               for t_score in t_scores]

    # setup progress bar
    idx_iter, total_iter = 0, sum(cfg.num_iterations)
    progress = ProgressBar(total_iter, num_increments=32)
    
    # for each octave
    for octave in range(cfg.num_octaves):
        
        # after first octave, resize images upward
        if octave>0:
            hi = octaves[-octave]
            img = hi + deepdream.resize(img, hi.shape[:2])
            
        if cfg.masks is not None:
            hw = img.shape[:2]
            mask = deepdream.resize(masks_orig, hw)

        # setup laplacian normalization (optional)
        if cfg.lap_n[octave] > 0:
            lap_norm_func = deepdream.laplacian_normalization(cfg.lap_n[octave])
        
        # octave optimization loop
        for i in range(cfg.num_iterations[octave]):
            
            gradient = np.zeros(img.shape).astype(np.float32)
            
            # do for each individual objective
            for t, t_grad in enumerate(t_grads):
            
                if cfg.masks is not None:
                    channel_mask = np.expand_dims(mask[:,:,t], -1)
                    if np.sum(channel_mask) == 0:
                        continue
                        
                # calculate plain gradient for channel objective    
                local_gradient = deepdream.calc_grad_tiled(img, t_grad)

                # smooth out high frequencies in gradient with Laplacian decomposition
                if cfg.lap_n[octave] > 0:
                    local_gradient = lap_norm_func(local_gradient)

                # "normalize" gradients by dividing by the mean
                if cfg.normalize_gradients:
                    abs_grad_mean = np.abs(local_gradient).mean()+1e-7
                    local_gradient = local_gradient / abs_grad_mean
                
                # multiply gradient by mask 
                if cfg.masks is not None:
                    local_gradient *= channel_mask
                    
                # force gradients grayscale
                if cfg.grayscale_gradients:
                    mean_grad = np.mean(local_gradient, 2)
                    local_gradient[:,:,:] = np.expand_dims(mean_grad, 2)
                    
                # add local gradient to gloab gradient
                gradient += local_gradient

            # finally add gglobal radient to image
            img += cfg.step[octave] * gradient
        
            # update console
            idx_iter += 1
            title = '%s: '%title if title is not None else ''
            update_str = '%sOctave %d/%d, Iter %d/%d' % (title, octave+1, cfg.num_octaves, idx_iter, total_iter)
            progress.update(update_str)
        
    # clip final image
    img = np.clip(img, 0, 255)
    
    IPython.display.clear_output()
    return img

