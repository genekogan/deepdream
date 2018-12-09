from mask import *

class Sequence:
    
    def __init__(self, attributes):
        self.sequence = []
        self.injections = []
        self.sy, self.sx = attributes['h'], attributes['w']
        self.oct_n, self.oct_s = attributes['oct_n'], attributes['oct_s']
        self.num_loop_frames = 0
        self.meta = None
        self.t = 0

    def get_channels(self):
        all_channels = [c for s in self.sequence for c in s['channels']]
        return all_channels

    def get_num_frames(self):
        return self.sequence[-1]['fade_out'][1]

    def get_num_loop_frames(self):
        return self.num_loop_frames

    def get_sequences_json(self):
        seq_json = copy.deepcopy(self.sequence)
        for seq in seq_json:
            seq['mask_constant'] = True if 'mask_constant' in seq else False
        return seq_json

    def append_json(self, seq_json):
        num_frames = seq_json['time'][1]-seq_json['time'][0]
        self.sequence.append(seq_json)
        self.t += num_frames

    def append(self, channels, mask, canvas, num_frames, blend_frames=0):
        mask_sizes = get_mask_sizes([self.sy, self.sx], self.oct_n, self.oct_s)
        new_sequence = {'mask':mask, 'channels':channels, 'canvas':canvas,
                        'fade_in': [self.t-blend_frames, self.t],
                        'time': [self.t, self.t+num_frames],
                        'fade_out': [self.t+num_frames, self.t+num_frames]}


        # hack -- just do image mask once (better would be to make it fully deterministic or match previous)
        # if mask['type'] == 'image':
        #     new_sequence['mask_constant'], meta = get_mask(mask, self.sy, self.sx, 0)





        if len(self.sequence) > 0 and blend_frames > 0:
            self.sequence[-1]['time'][1] = self.t-blend_frames
            self.sequence[-1]['fade_out'] = [self.t-blend_frames, self.t]
        self.sequence.append(new_sequence)
        self.t += num_frames

    def get_frame(self, t):
        t = t % self.get_num_frames()
        canvas, components = None, []
        print("------- mask ------ ", t)
        
        for s, seq in enumerate(self.sequence):
            (t01,t02), (t11,t12), (t21, t22), weight = seq['fade_in'], seq['time'], seq['fade_out'], 0
            if   t01<=t<t02: weight = float(t-t01)/(t02-t01)
            elif t11<=t<t12: weight = 1.0
            elif t21<=t<t22: weight = abs(float(t22-t))/(t22-t21)
            if   t11<=t<t22: canvas = seq['canvas']
            
            
            #print(" ",s,t,t11,t-t11, t-t01)
            #masks, meta = get_mask(seq['mask'], self.sy, self.sx, t-t11, self.meta)
            masks, meta = get_mask(seq['mask'], self.sy, self.sx, t-t01, self.meta)
            
            #masks, self.meta = get_mask(seq['mask'], self.sy, self.sx, t-t11) if 'mask_constant' not in seq else seq['mask_constant']
            


            #self.meta = meta            

            # match_prev_masks = True
            # if seq['match_prev']:
            #     # adjust masks to prev_mask 
            #     seq['mask']['prev_mask'] = masks

            # make masks prevMasks



            components += [{"mask":masks[:,:,c], "channel":seq['channels'][c], "weight":weight} for c in range(seq['mask']['n'])]
        mask, channels = np.zeros((self.sy, self.sx, 0)), []
        for component in components:
            if component['channel'] not in channels:
                channels.append(component['channel'])
                mask = np.append(mask, np.zeros((self.sy, self.sx, 1)), axis=2)
            idx = channels.index(component['channel'])
            mask[:, :, idx] += (component['mask'] * component['weight'])
        injections = [{'path':inj['path'], 'amt':inj['amt'][t-inj['time'][0]]} 
                      for inj in self.injections if inj['time'][0]<=t<inj['time'][1]]
        injection = None if len(injections)==0 else injections[0]
        return mask, channels, canvas, injection
    
    def add_injection(self, path, t1, t2, top):
        n = t2-t1
        amt = [top*2*((n-1)/2-abs(i-(n-1)/2))/(n-1) for i in range(n)]
        self.injections.append({'path':path, 'time':[t1,t2], 'amt':amt})
       
    def loop_to_beginning(self, num_loop_frames):
        self.num_loop_frames = num_loop_frames


        
def view_sequence_mask(sequence, attributes, output, flatten_blend=False, draw_rgb=False, animate=False):
    name, save_all, save_last = output['name'], output['save_all'], output['save_last']
    sy, sx, iter_n, step, oct_n, oct_s, lap_n = attributes['h'], attributes['w'], attributes['iter_n'], attributes['step'], attributes['oct_n'], attributes['oct_s'], attributes['lap_n']
    mask_sizes = get_mask_sizes([sy, sx], oct_n, oct_s)
    num_frames, num_loop_frames = sequence.get_num_frames(), sequence.get_num_loop_frames()
    for f in range(num_frames):
        mask, channels, canvas, injection = sequence.get_frame(f)
        draw_mask(mask, flatten_blend=flatten_blend, draw_rgb=draw_rgb, animate=animate, color_labels=None)