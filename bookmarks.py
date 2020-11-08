from random import random


def get_list_favorites(layers):
    favorites = []
    for l in layers:
        for c in l["channels"]:
            full_name = layer_lookup[l["layer"]]
            favorites.append({"layer":l["layer"], "channel":c})
            #favorites.append({"nickname":l["layer"], "layer":full_name, "channel":c})
    return favorites
           

def get_random_favorites(favorite_layers, n):
    favorites = get_list_favorites(favorite_layers)
    list_good = False
    while not list_good:
        indexes = [int(len(favorites) * random()) for i in range(n)]
        list_good = (len(indexes) == len(set(indexes)))  # make sure unique
    channels = [favorites[idx] for idx in indexes]
    channels = [{'layer': layer_lookup[c['layer']], 
                 'channel': c['channel']} 
                for c in channels]
    return channels


def get_random(whichlayer, n):
    layer_name = layer_lookup[whichlayer]
    num_channels = int(T(layer_name).shape[3])
    idx_channels = [int(num_channels * random()) for i in range(n)]
    channels = [{"layer":whichlayer, "channel":c} for c in idx_channels]
    return channels


def get_bookmarks_via_alias(layer_alias):
    return layer_aliases[layer_alias]


layer_lookup={
    'A1': 'conv2d0_pre_relu',
    'A2': 'conv2d1_pre_relu',
    'A3': 'conv2d2_pre_relu',
    'B1': 'mixed3a_1x1_pre_relu',
    'B2': 'mixed3a_3x3_bottleneck_pre_relu',
    'B3': 'mixed3a_3x3_pre_relu',
    'B4': 'mixed3a_5x5_bottleneck_pre_relu',
    'B5': 'mixed3a_5x5_pre_relu',
    'B6': 'mixed3a_pool_reduce_pre_relu',
    'C1': 'mixed3b_1x1_pre_relu',
    'C2': 'mixed3b_3x3_bottleneck_pre_relu',
    'C3': 'mixed3b_3x3_pre_relu',
    'C4': 'mixed3b_5x5_bottleneck_pre_relu',
    'C5': 'mixed3b_5x5_pre_relu',
    'C6': 'mixed3b_pool_reduce_pre_relu',
    'D1': 'mixed4a_1x1_pre_relu',
    'D2': 'mixed4a_3x3_bottleneck_pre_relu',
    'D3': 'mixed4a_3x3_pre_relu',
    'D4': 'mixed4a_5x5_bottleneck_pre_relu',
    'D5': 'mixed4a_5x5_pre_relu',
    'D6': 'mixed4a_pool_reduce_pre_relu',
    'E1': 'mixed4b_1x1_pre_relu',
    'E2': 'mixed4b_3x3_bottleneck_pre_relu',
    'E3': 'mixed4b_3x3_pre_relu',
    'E4': 'mixed4b_5x5_bottleneck_pre_relu',
    'E5': 'mixed4b_5x5_pre_relu',
    'E6': 'mixed4b_pool_reduce_pre_relu',
    'F1': 'mixed4c_1x1_pre_relu',
    'F2': 'mixed4c_3x3_bottleneck_pre_relu',
    'F3': 'mixed4c_3x3_pre_relu',
    'F4': 'mixed4c_5x5_bottleneck_pre_relu',
    'F5': 'mixed4c_5x5_pre_relu',
    'F6': 'mixed4c_pool_reduce_pre_relu',
    'G1': 'mixed4d_1x1_pre_relu',
    'G2': 'mixed4d_3x3_bottleneck_pre_relu',
    'G3': 'mixed4d_3x3_pre_relu',
    'G4': 'mixed4d_5x5_bottleneck_pre_relu',
    'G5': 'mixed4d_5x5_pre_relu',
    'G6': 'mixed4d_pool_reduce_pre_relu',
    'H1': 'mixed4e_1x1_pre_relu',
    'H2': 'mixed4e_3x3_bottleneck_pre_relu',
    'H3': 'mixed4e_3x3_pre_relu',
    'H4': 'mixed4e_5x5_bottleneck_pre_relu',
    'H5': 'mixed4e_5x5_pre_relu',
    'H6': 'mixed4e_pool_reduce_pre_relu',
    'J1': 'mixed5a_1x1_pre_relu',
    'J2': 'mixed5a_3x3_bottleneck_pre_relu',
    'J3': 'mixed5a_3x3_pre_relu',
    'J4': 'mixed5a_5x5_bottleneck_pre_relu',
    'J5': 'mixed5a_5x5_pre_relu',
    'J6': 'mixed5a_pool_reduce_pre_relu',
    'K1': 'mixed5b_1x1_pre_relu',
    'K2': 'mixed5b_3x3_bottleneck_pre_relu',
    'K3': 'mixed5b_3x3_pre_relu',
    'K4': 'mixed5b_5x5_bottleneck_pre_relu',
    'K5': 'mixed5b_5x5_pre_relu',
    'K6': 'mixed5b_pool_reduce_pre_relu',
    'L1': 'head0_bottleneck_pre_relu',
    'L2': 'head1_bottleneck_pre_relu'
}

layers_c1 = [   
    {"layer":'A1', "channels":[7,15,1,28,52,8]},
    {"layer":'A2', "channels":[12,3,38,25,27,18,29,14]},
    {"layer":'A3', "channels":[159,30,20,170,130,167,89,150,28,124,31,58,19,75,69]}
]

layers_m3 = [
    {"layer":'B1', "channels":[5,55,21,38,11,60,42,61,58,39,37,8,31,20,52,9]},
    {"layer":'B2', "channels":[3,6,42,57,54,26,44,40,10,19,0,30,36,45,38]},
    {"layer":'B3', "channels":[15,13,7,43,57,4,27,56,61,62,54,51,39,64,76,113]},
    {"layer":'B4', "channels":[10,4,2,7]},
    {"layer":'B5', "channels":[12,2]},
    {"layer":'B6', "channels":[13,14,15,6,7]},

    {"layer":'C1', "channels":[7,14,2,5,10,9,1,33,46,81,106,108,95,110,127]},
    {"layer":'C2', "channels":[13,6,10,0,8,44]},
    {"layer":'C3', "channels":[10,0,3,12,6,1,11,7,123,128,179,183,188]},
    {"layer":'C4', "channels":[10,1,11,14,15]},
    {"layer":'C5', "channels":[13,1,4,9,5,0,3,39,53,87,88]},
    {"layer":'C6', "channels":[5,15,6,9,4,8,3,2,19,30,35,38,54]}
]

layers_m4 = [
    {"layer":'D1', "channels":[36,63,46,144,178,138,35,1,43,41,96,30,51,97,121,127,149,54,101,120,88,4,3,170,158,81,68,131,55,110,162,137,86,118,85,21,189,191,155,14,172,133,20,57,2,67,38,130,186,112,164,176,11,27,117,103,163,32,87,152]},
    {"layer":'D2', "channels":[53,92,42,84,6,50,24,55,91,59,38,8,35,20,73,13]},
    {"layer":'D3', "channels":[68,4,47,21,32,1,82,85,48,27,65,92,33,100,111,118,125,130,138,139,147,174,185,201]},
    {"layer":'D4', "channels":[5]},
    {"layer":'D5', "channels":[15]},
    {"layer":'D6', "channels":[14,9,12,3,11,15,20,31,32,52]},

    {"layer":'E1', "channels":[58,23,42,125,6,100,62,31,115,68]},
    {"layer":'E2', "channels":[91,45,81,7,109,85,52,62,78,37]},
    {"layer":'E3', "channels":[95,78,59,36,82,17,50,15,127,130,150,172,188,189,196]},
    {"layer":'E4', "channels":[15,14,21]},
    {"layer":'E5', "channels":[3]},
    {"layer":'E6', "channels":[0,22,14,12,1,16,17,21,27,33,37,38,44,50,63]},

    {"layer":'F1', "channels":[13,22,30,31,40,49,78,83,84,94]},
    {"layer":'F2', "channels":[6,7,8,18,32,41,64,79,80,85,120,126]},
    {"layer":'F3', "channels":[48,75,33,124,10,46,95,107,26,109,147,164,187,230,245,254]},
    {"layer":'F4', "channels":[20,21]},
    {"layer":'F5', "channels":[1,6,7,11,14,17,24,28,33,40,44,46,50,53,56,60,63]},
    {"layer":'F6', "channels":[9,14,17,21,29,30,34,41,42,43,45,53,54,56,61,62]},
    
    {"layer":'G1', "channels":[9,10,15,23,34,36,39,42,44,62,72,73,76,85,90,91,94,107,109]},
    {"layer":'G2', "channels":[1,7,12,14,15,16,22,30,36,38,42,54,66,76,84,86,89,91,98,120,130,136,139,142]},
    {"layer":'G3', "channels":[11,23,33,45,52,67,90,95,105,123,145,185,191,199,232,248,283]},
    {"layer":'G4', "channels":[1,7]},
    {"layer":'G5', "channels":[3,14,21,56]},
    {"layer":'G6', "channels":[2,3,5,10,11,14,15,20,25,30,31,34,36,40,47,49,52,58]},
    
    {"layer":'H1', "channels":[1,5,11,22,34,45,54,57,55,66,93,99,102,107,116,125,155,158,174,183,219,225,250,255]},
    {"layer":'H2', "channels":[4,38,78,101]},
    {"layer":'H3', "channels":[5,7,8,18,42,43,59,77,79,103,136,144,149,192,200,230,233,268,270,282,303]},
    {"layer":'H4', "channels":[8,13]},
    {"layer":'H5', "channels":[9,20,26,34,37,43,67,89]},
    {"layer":'H6', "channels":[1,5,7,23,26,34,38,51,68,70,71,72,75,77,83,88,93,96,99,101,109,110,112,123,124]}
]

layers_m5 = [
    {"layer":'J1', "channels":[13,19,29,42,49,51,64,75,84,102,110,114,132,150,168,183,189,198,222,243,244]},
    {"layer":'J2', "channels":[9,18,32,48,69,159]},
    {"layer":'J3', "channels":[23,25,38,50,93,171,180,183,223,265,316]},
    {"layer":'J4', "channels":[8,21,42]},
    {"layer":'J5', "channels":[4,42,54,81,85,87,110,125]},
    {"layer":'J6', "channels":[16,17,53,54,65,71,75,82,89,103,109,120]},

    {"layer":'K1', "channels":[6,27,41,42,134,174,179,195,217,221,318,382]},
    {"layer":'K2', "channels":[1,7,9,22,33,52,87,101,109,113,146,155,189]},
    {"layer":'K3', "channels":[96,98,101,109,126,187,238,318,331]},
    {"layer":'K4', "channels":[2,12,41,46]},
    {"layer":'K5', "channels":[3,14,72,112]},
    {"layer":'K6', "channels":[17,41,80]}
]

layers_h1 = [
    {"layer":'L1', "channels":[50,97,46,125,18,126,112,9,114,35,20,28,10,60,119,124,48,108,67,11,86,87,110,59,103,127,69,91,44,21,56,41,55,96,64,89,92,83,17,]},
    {"layer":'L2', "channels":[65,96,125,20,81,15,84,31,1,54]}
]


faves_c1 = [
    {"layer":"A3", "channels":[89]}
]

faves_m3 = [
    {"layer":"B1", "channels":[5,31]},
    {"layer":"B2", "channels":[10,44]},
    {"layer":"B3", "channels":[27,62,76]},
    {"layer":"B4", "channels":[10]},
    {"layer":"B6", "channels":[15]},
    {"layer":"C1", "channels":[1,7,81,106]},
    {"layer":"C2", "channels":[0,44]},
    {"layer":"C3", "channels":[1,128,179]},
    {"layer":"C5", "channels":[39]},
    {"layer":"C6", "channels":[5,30,38,54]}
]

faves_m4 = [
    {"layer":"D1", "channels":[27,41,67,86,96,101,112,120,133,144,152,162,163,186,189]},
    {"layer":"D2", "channels":[13,50]},
    {"layer":"D3", "channels":[27,32,92,100]},
    {"layer":"D5", "channels":[15]},
    {"layer":"D6", "channels":[9,11,12,32]},
    {"layer":"E1", "channels":[100,115]},
    {"layer":"E2", "channels":[62]},
    {"layer":"F1", "channels":[40]},
    {"layer":"F2", "channels":[41]},
    {"layer":"F3", "channels":[147,187,254]},
    {"layer":"F5", "channels":[40,44,50,63]},
    {"layer":"F6", "channels":[9,17,34,41,42,54]},
    {"layer":"G1", "channels":[15,36,62,76,94]},
    {"layer":"G2", "channels":[1,7,15,30,36,42,54,91,136]},
    {"layer":"G6", "channels":[5,31,47,49,58]},
    {"layer":"H1", "channels":[5,22,34,102,107,116,158,183,225]},
    {"layer":"H2", "channels":[4,38]},
    {"layer":"H3", "channels":[7,77,103,136,303]},
    {"layer":"H4", "channels":[13]},
    {"layer":"H5", "channels":[34]},
    {"layer":"H6", "channels":[26,72,88,112,123]}
]

faves_m5 = [
    {"layer":"J1", "channels":[13,42,75,84,114]},
    {"layer":"J3", "channels":[25,223,265]},
    {"layer":"J5", "channels":[81]},
    {"layer":"J6", "channels":[54,103]},
    {"layer":"K1", "channels":[42,179]},
    {"layer":"K2", "channels":[1,113,146,126]},
    {"layer":"K5", "channels":[72]},
    {"layer":"K6", "channels":[41]}
]

faves_h1 = [
    {"layer":"L1", "channels":[9,18,20,28,35,41,46,55,86,97,103,108,124]}
]

birds=[{"layer":"F6","channels":[34]},{"layer":"G6","channels":[2,11,15]},{"layer":"H1","channels":[250]},{"layer":"H6","channels":[75,93]},{"layer":"J3","channels":[50]},{"layer":"J6","channels":[16]},{"layer":"L1","channels":[10]}]

bowl=[{"layer":"D1","channels":[96]},{"layer":"E1","channels":[62]},{"layer":"E2","channels":[78]},{"layer":"E3","channels":[150]},{"layer":"F2","channels":[32]},{"layer":"G2","channels":[38]},{"layer":"G3","channels":[185]},{"layer":"G6","channels":[14]},{"layer":"J3","channels":[171]},{"layer":"L1","channels":[56]}]

brass=[{"layer":"F6","channels":[56]},{"layer":"H1","channels":[1]},{"layer":"H6","channels":[124]},{"layer":"K2","channels":[7]},{"layer":"L1","channels":[91]}]

corner=[{"layer":"B2","channels":[44]},{"layer":"B6","channels":[15]},{"layer":"C6","channels":[54]},{"layer":"D1","channels":[41,101,120]},{"layer":"D3","channels":[92]},{"layer":"D5","channels":[15]},{"layer":"D6","channels":[11,12,52]},{"layer":"E6","channels":[21]},{"layer":"F1","channels":[78]},{"layer":"F1","channels":[94]}]

dof=[{"layer":"F2","channels":[7]},{"layer":"H2","channels":[38]},{"layer":"L1","channels":[48]}]

dogs=[{"layer":"G2","channels":[1]},{"layer":"G3","channels":[232]}]

eyes=[{"layer":"F3","channels":[48]},{"layer":"F5","channels":[7,24]},{"layer":"G2","channels":[1]},{"layer":"H1","channels":[250]}]

facade=[{"layer":"D1","channels":[130]},{"layer":"D2","channels":[6]},{"layer":"E3","channels":[95,130]},{"layer":"E6","channels":[33,44]},{"layer":"F2","channels":[6,64]},{"layer":"F3","channels":[75,95,107]},{"layer":"F6","channels":[29,61]},{"layer":"G2","channels":[66]},{"layer":"G6","channels":[5,10,25,49]},{"layer":"H1","channels":[158]},{"layer":"H3","channels":[77]},{"layer":"H6","channels":[5]},{"layer":"J1","channels":[49]},{"layer":"L1","channels":[59]}]

flower=[{"layer":"E3","channels":[17,36]},{"layer":"E6","channels":[16]},{"layer":"F2","channels":[120]},{"layer":"F3","channels":[230]},{"layer":"F6","channels":[17]},{"layer":"G2","channels":[139]},{"layer":"G3","channels":[145]},{"layer":"G5","channels":[21]},{"layer":"H3","channels":[43,136,192]}]

food=[{"layer":"E3","channels":[188]},{"layer":"F1","channels":[30]},{"layer":"F3","channels":[26]},{"layer":"F3","channels":[109]},{"layer":"G1","channels":[76]},{"layer":"G2","channels":[76,142]},{"layer":"G3","channels":[52,90,283]},{"layer":"G6","channels":[52]},{"layer":"H1","channels":[54]},{"layer":"H6","channels":[7]},{"layer":"J1","channels":[183,244]},{"layer":"L1","channels":[44,55]},{"layer":"L2","channels":[15]}]

nature=[{"layer":"D2","channels":[92]},{"layer":"D3","channels":[118,147]},{"layer":"D6","channels":[31]},{"layer":"E1","channels":[58]},{"layer":"E6","channels":[17]},{"layer":"F1","channels":[40]},{"layer":"F4","channels":[20]},{"layer":"F6","channels":[14]},{"layer":"G2","channels":[42]},{"layer":"H1","channels":[116]},{"layer":"H2","channels":[4]},{"layer":"H3","channels":[8,233,282]},{"layer":"H5","channels":[9]},{"layer":"H6","channels":[34,51]},{"layer":"J1","channels":[51,198]},{"layer":"J5","channels":[42]},{"layer":"J6","channels":[65]},{"layer":"J6","channels":[103]}]

pattern=[{"layer":"C6","channels":[38]},{"layer":"D6","channels":[32]},{"layer":"E6","channels":[63]},{"layer":"F5","channels":[33]},{"layer":"F5","channels":[56]},{"layer":"G2","channels":[86]}]

pipe=[{"layer":"E6","channels":[0]},{"layer":"F6","channels":[54]},{"layer":"G1","channels":[34,85]},{"layer":"L1","channels":[89,92]}]

sea=[{"layer":"E3","channels":[15]},{"layer":"F6","channels":[53]},{"layer":"F6","channels":[62]},{"layer":"G1","channels":[9,36]},{"layer":"G2","channels":[16,89]},{"layer":"G3","channels":[45,105]},{"layer":"H1","channels":[22,107]},{"layer":"H6","channels":[23,70,83]},{"layer":"J6","channels":[109]},{"layer":"L1","channels":[83]}]

simple=[{"layer":"A1","channels":[7,52]},{"layer":"A2","channels":[18,25]},{"layer":"A3","channels":[19,30,89,150,159,170]},{"layer":"B1","channels":[5,8]},{"layer":"B3","channels":[7,13,27]}]

stones=[{"layer":"G2","channels":[54]},{"layer":"H6","channels":[72]}]

thread=[{"layer":"B1","channels":[31]},{"layer":"B6","channels":[6]},{"layer":"C1","channels":[33]},{"layer":"C2","channels":[0,44]},{"layer":"D1","channels":[67]},{"layer":"D2","channels":[55]},{"layer":"D3","channels":[174,185,201]},{"layer":"E1","channels":[115,125]},{"layer":"E2","channels":[52]},{"layer":"E6","channels":[38]},{"layer":"F3","channels":[254]},{"layer":"G3","channels":[248]},{"layer":"H1","channels":[66]},{"layer":"H6","channels":[77,101,123]},{"layer":"J2","channels":[32]},{"layer":"L1","channels":[48,119,125,126]}]

wheel=[{"layer":"G6","channels":[20]},{"layer":"H5","channels":[20,89]}]

fave_moving1 = [
    {"layer":'A1', "channels":[15,52,7,8]},
    {"layer":'A2', "channels":[14,29]},
    {"layer":'A3', "channels":[124,130,150,170,19,30,31,58,75]},
    {"layer":'B1', "channels":[31,39,42]},
    {"layer":'B2', "channels":[10]},
    {"layer":'B3', "channels":[51,76]},
    {"layer":'C1', "channels":[7,81]},
    {"layer":'C2', "channels":[6]},
    {"layer":'C5', "channels":[39]},
    {"layer":'C6', "channels":[38,5,54]},
    {"layer":'D1', "channels":[120,41,63,96]},
    {"layer":'D2', "channels":[50]},
    {"layer":'D3', "channels":[130,185,27,92]},
    {"layer":'D5', "channels":[15]},
    {"layer":'D6', "channels":[27,31,32]},
    {"layer":'E2', "channels":[62]},
    {"layer":'E6', "channels":[27]},
    {"layer":'F1', "channels":[40]},
    {"layer":'F3', "channels":[187]},
    {"layer":'F5', "channels":[63]},
    {"layer":'F6', "channels":[9]},
    {"layer":'G2', "channels":[130,139,30,7,84]},
    {"layer":'G3', "channels":[232]},
    {"layer":'G6', "channels":[40]},
    {"layer":'H1', "channels":[107,22]},
    {"layer":'H2', "channels":[38]},
    {"layer":'H5', "channels":[37]},
    {"layer":'H6', "channels":[88]},
    {"layer":'J1', "channels":[102]}
]

fave_moving2 = [
    {"layer":'D3', "channels":[27,85]},
    {"layer":'D6', "channels":[9,52]},
    {"layer":'E1', "channels":[125]},
    {"layer":'E3', "channels":[17 ]},
    {"layer":'F1', "channels":[31,49,78]},
    {"layer":'F2', "channels":[64]},
    {"layer":'F3', "channels":[245]},
    {"layer":'G2', "channels":[1,16]},
    {"layer":'G3', "channels":[283]}
]

fave_moving = [
    {"layer":'A1', "channels":[15,52,7,8]},
    {"layer":'A2', "channels":[14,29]},
    {"layer":'A3', "channels":[124,130,150,170,19,30,31,58,75]},
    {"layer":'B1', "channels":[31,39,42,5]},
    {"layer":'B2', "channels":[10]},
    {"layer":'B3', "channels":[51,76]},
    {"layer":'C1', "channels":[7,81,77]},
    {"layer":'C2', "channels":[6,44]},
    {"layer":'C3', "channels":[179]},
    {"layer":'C4', "channels":[10]},
    {"layer":'C5', "channels":[39]},
    {"layer":'C6', "channels":[30,38,5,54]},
    {"layer":'D1', "channels":[120,144,41,63,96]},
    {"layer":'D2', "channels":[50]},
    {"layer":'D3', "channels":[85,100,33,130,185,27,92]},
    {"layer":'D5', "channels":[15]},
    {"layer":'D6', "channels":[9,52,27,31,32]},
    {"layer":'E1', "channels":[125,115,58]},
    {"layer":'E2', "channels":[62]},
    {"layer":'E3', "channels":[17]},
    {"layer":'E4', "channels":[15]},
    {"layer":'E6', "channels":[27,0]},
    {"layer":'F1', "channels":[40,31,49,78]},
    {"layer":'F2', "channels":[64,41]},
    {"layer":'F3', "channels":[187,245,254]},
    {"layer":'F5', "channels":[63,63]},
    {"layer":'F6', "channels":[9,56,41]},
    {"layer":'G1', "channels":[62,34]},
    {"layer":'G2', "channels":[1,16,130,139,30,7,84]},
    {"layer":'G3', "channels":[283,232,52]},
    {"layer":'G6', "channels":[40]},
    {"layer":'H1', "channels":[107,22,225]},
    {"layer":'H2', "channels":[38]},
    {"layer":'H3', "channels":[77,79]},
    {"layer":'H5', "channels":[37]},
    {"layer":'H6', "channels":[88]},
    {"layer":'J1', "channels":[102,42]},
    {"layer":'J2', "channels":[48]},
    {"layer":'J6', "channels":[65]}
]


layer_aliases = {}
layer_aliases['layers_c1']=layers_c1
layer_aliases['layers_m3']=layers_m3
layer_aliases['layers_m4']=layers_m4
layer_aliases['layers_m5']=layers_m5
layer_aliases['layers_h1']=layers_h1
layer_aliases['faves_c1']=faves_c1
layer_aliases['faves_m3']=faves_m3
layer_aliases['faves_m4']=faves_m4
layer_aliases['faves_m5']=faves_m5
layer_aliases['faves_h1']=faves_h1
layer_aliases['birds']=birds
layer_aliases['bowl']=bowl
layer_aliases['brass']=brass
layer_aliases['corner']=corner
layer_aliases['dof']=dof
layer_aliases['dogs']=dogs
layer_aliases['eyes']=eyes
layer_aliases['facade']=facade
layer_aliases['flower']=flower
layer_aliases['food']=food
layer_aliases['nature']=nature
layer_aliases['pattern']=pattern
layer_aliases['pipe']=pipe
layer_aliases['sea']=sea
layer_aliases['simple']=simple
layer_aliases['stones']=stones
layer_aliases['thread']=thread
layer_aliases['wheel']=wheel
layer_aliases['fave_moving2']=fave_moving2
layer_aliases['fave_moving1']=fave_moving1
layer_aliases['fave_moving']=fave_moving


