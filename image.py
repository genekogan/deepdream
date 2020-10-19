import IPython
import os
from PIL import Image
import numpy as np
import requests
import urllib.parse
from io import BytesIO



class EasyDict(dict):
    def __init__(self, *args, **kwargs):
        super(EasyDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class ProgressBar:
    
    def __init__(self, total_iter, num_increments=32):
        self.num_increments = num_increments
        self.idx_iter, self.total_iter = 0, total_iter
        self.iter_per = self.total_iter / self.num_increments

    def update(self, update_str=''):
        self.idx_iter += 1
        progress_iter = int(self.idx_iter / self.iter_per)
        progress_str  = '[' + '=' * progress_iter 
        progress_str += '-' * (self.num_increments - progress_iter) + ']'
        IPython.display.clear_output(wait=True)
        IPython.display.display(progress_str+'  '+update_str)

    
def load_image(image, image_size=None, to_numpy=False, normalize=False):
    if isinstance(image, str):
        if is_url(image):
            image = url_to_image(image)
        elif os.path.exists(image):
            image = Image.open(image).convert('RGB')
        else:
            raise ValueError('no image found at %s'%image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8)).convert('RGB')
    if image_size is not None and isinstance(image_size, tuple):
        image = resize(image, image_size)
    elif image_size is not None and not isinstance(image_size, tuple):
        aspect = get_aspect_ratio(image)
        image_size = (int(aspect * image_size), image_size)
        image = resize(image, image_size)
    if to_numpy:
        image = np.array(image)
        if normalize:
            image = image / 255.0        
    return image


def get_size(image):
    if isinstance(image, str):
        image = load_image(image, 1024)
        w, h = image.size
    elif isinstance(image, Image.Image):    
        w, h = image.size
    elif isinstance(image, np.ndarray):
        w, h = image.shape[1], image.shape[0]
    return w, h


def get_aspect_ratio(image):
    w, h = get_size(image)
    return float(w) / h


def resize(img, new_size, mode=None, align_corners=True):
    sampling_modes = {
        'nearest': Image.NEAREST, 
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC, 
        'lanczos': Image.LANCZOS
    }
    assert isinstance(new_size, tuple), \
        'Error: image_size must be a tuple.'
    assert mode is None or mode in sampling_modes.keys(), \
        'Error: resample mode %s not understood: options are nearest, bilinear, bicubic, lanczos.'
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
    w1, h1 = img.size
    w2, h2 = new_size
    if (h1, w1) == (w2, h2):
        return img
    if mode is None:
        mode = 'bicubic' if w2*h2 >= w1*h1 else 'lanczos'
    resample_mode = sampling_modes[mode]
    return img.resize((w2, h2), resample=resample_mode)


def url_to_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img


def is_url(path):
    path = urllib.parse.urlparse(path)
    return path.netloc != ''
   

def save(img, filename):
    folder = os.path.dirname(filename)
    if folder and not os.path.isdir(folder):
        os.mkdir(folder)
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
    img.save(str(filename))

    
def display(img):
    if isinstance(img, list):
        return frames_to_movie(img, fps=30)
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
    IPython.display.display(img)
