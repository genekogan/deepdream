from functools import partial
import numpy as np
import tensorflow as tf




class DeepDreamArgs:
    
    def __init__(self):
        self.tile_size = 512
        self.model_file = 'model/tensorflow_inception_graph.pb'
    
    def __str__(self):
        args = [
            'tile_size: %d' % self.tile_size,
            'model_file: %s' % self.model_file
        ]
        return ', '.join(args)


    

class DeepDream():
    
    def __init__(self, params):
        self.is_setup = False
        self.params = params
        self.setup(self.params.model_file)

    
    def setup(self, model_fn):
        if self.is_setup:
            return
        
        self.graph = tf.Graph()
        self.sess = tf.InteractiveSession(graph=self.graph)

        with tf.gfile.FastGFile(model_fn, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        self.t_input = tf.placeholder(np.float32, name='input')
        imagenet_mean = 117.0
        t_preprocessed = tf.expand_dims(self.t_input-imagenet_mean, 0)
        tf.import_graph_def(graph_def, {'input':t_preprocessed})

        
        self.print_layers()
        
        
        self.resize = tffunc(self.sess, np.float32, np.int32)(self.resize_helper)
        
        k = np.float32([1,4,6,4,1])
        k = np.outer(k, k)
        self.k5x5 = k[:,:,None,None]/k.sum()*np.eye(3, dtype=np.float32)

        self.is_setup = True
        
        
    def print_layers(self):
        layers = [op.name for op in self.graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
        feature_nums = [int(self.graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]
        print('Number of layers', len(layers))
        print('Total number of feature channels:', sum(feature_nums))
        for layer, feature_num in zip(layers, feature_nums):
            print(' %s (%d)' % (layer, feature_num))
        
    def laplacian_normalization(self, lap_n):
        return tffunc(self.sess, np.float32)(partial(
            self.lap_normalize, 
            scale_n=lap_n))
    

    def T(self, layer):
        '''Helper for getting layer output tensor'''
        return self.graph.get_tensor_by_name("import/%s:0"%layer)
        
        
    def resize_helper(self, img, size):
        '''Resize tensor with bilinear interpolation'''
        img = tf.expand_dims(img, 0)
        return tf.image.resize_bilinear(img, size)[0,:,:,:]
    
    def calc_grad_tiled(self, img, t_grad):
        '''Compute the value of tensor t_grad over the image in a tiled way.
        Random shifts are applied to the image to blur tile boundaries over 
        multiple iterations.'''
        sz = self.params.tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        for y in range(0, max(h-sz//2, sz),sz):
            for x in range(0, max(w-sz//2, sz),sz):
                sub = img_shift[y:y+sz,x:x+sz]
                g = self.sess.run(t_grad, {self.t_input:sub})
                grad[y:y+sz,x:x+sz] = g
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)

    

    


    def lap_split(self, img):
        '''Split the image into lo and hi frequency components'''
        with tf.name_scope('split'):
            lo = tf.nn.conv2d(img, self.k5x5, [1,2,2,1], 'SAME')
            lo2 = tf.nn.conv2d_transpose(lo, self.k5x5*4, tf.shape(img), [1,2,2,1])
            hi = img-lo2
        return lo, hi

    def lap_split_n(self, img, n):
        '''Build Laplacian pyramid with n splits'''
        levels = []
        for i in range(n):
            img, hi = self.lap_split(img)
            levels.append(hi)
        levels.append(img)
        return levels[::-1]

    def lap_merge(self, levels):
        '''Merge Laplacian pyramid'''
        img = levels[0]
        for hi in levels[1:]:
            with tf.name_scope('merge'):
                img = tf.nn.conv2d_transpose(img, self.k5x5*4, tf.shape(hi), [1,2,2,1]) + hi
        return img

    def normalize_std(self, img, eps=1e-10):
        '''Normalize image by making its standard deviation = 1.0'''
        with tf.name_scope('normalize'):
            std = tf.sqrt(tf.reduce_mean(tf.square(img)))
            return img/tf.maximum(std, eps)

    def lap_normalize(self, img, scale_n=4):
        '''Perform the Laplacian pyramid normalization.'''
        img = tf.expand_dims(img,0)
        tlevels = self.lap_split_n(img, scale_n)
        tlevels = list(map(self.normalize_std, tlevels))
        out = self.lap_merge(tlevels)
        return out[0,:,:,:]





def tffunc(session, *argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.'''
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=session)
        return wrapper
    return wrap
