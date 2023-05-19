"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
from tabnanny import check
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

import sys
sys.path.append('..')

import torch
import torch.nn as nn
# torch.manual_seed(0)
import scipy.misc
import json
from collections import OrderedDict
import numpy as np
import os
device_ids = [0]
from PIL import Image
import gc

import pickle
from models.stylegan1 import G_mapping,Truncation,G_synthesis
from models.stylegan2 import MappingNetwork, SynthesisNetwork

import copy
from numpy.random import choice
from torch.distributions import Categorical
import scipy.stats
from torch_utils.utils import multi_acc, colorize_mask, get_label_stas, oht_to_scalar, Interpolate, get_image_difference #latent_to_image
import torch.optim as optim
import argparse
import glob
from torch.utils.data import Dataset, DataLoader
device = 'cuda:' + str(device_ids[0]) if torch.cuda.is_available() else 'cpu'
print("Device:", device)
import dnnlib
from torch_utils import misc
from training_scripts_DB_SG2 import legacy
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


#import cv2
from PIL import Image
import imageio
from tqdm import tqdm 

from training_scripts_DB_SG2.network_preparation import prepare_SG2
from interpreter_utils.utils import latent_to_image, Interpolate

import time


class trainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class pixel_classifier(nn.Module):
    def __init__(self, numpy_class, dim):
        super(pixel_classifier, self).__init__()
        if numpy_class < 32:
            self.layers = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=32),
                nn.Linear(32, numpy_class),
                # nn.Sigmoid()
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, numpy_class),
                # nn.Sigmoid()
            )


    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)



    def forward(self, x):
        return self.layers(x)

def prepare_stylegan(args):

    ######################################################################################################
    # Code Disabled, we use stylegan2 
    if args['stylegan_ver'] == "1" and False:
        if args['category'] == "car":
            resolution = 512
            max_layer = 8
        elif  args['category'] == "face":
            resolution = 1024
            max_layer = 8
        elif args['category'] == "bedroom":
            resolution = 256
            max_layer = 7
        elif args['category'] == "cat":
            resolution = 256
            max_layer = 7
        elif args['category'] == "eyes_256":
            print("Eyes category")
            resolution = 256
            max_layer = 7

        elif args['category'] == "eyes_512":
            print("Eyes category")
            resolution = 512
            max_layer = 7
        else:
            assert "Not implementated!"

        print("---- Resolution:", resolution, " Layers:", max_layer)

        print("---- Get avg latent")
        avg_latent = np.load(args['average_latent'])
        print("---- Latent to torch")
        avg_latent = torch.from_numpy(avg_latent).type(torch.FloatTensor).to(device)
        print("AVG latent", avg_latent.shape) # SHOULD BE 18, 512
        print("---- Build Generator")

        
        g_all = nn.Sequential(OrderedDict([
            ('g_mapping', G_mapping()),
            ('truncation', Truncation(avg_latent,max_layer=max_layer, device=device, threshold=0.7)),
            ('g_synthesis', G_synthesis( resolution=resolution))
        ]))

        
        print("---- Load state dict")
        g_all.load_state_dict(torch.load(args['stylegan_checkpoint'], map_location=device))
        g_all.eval()

        print("---- Do parallel")
        g_all = nn.DataParallel(g_all, device_ids=device_ids).cuda()


    ######################################################################################################
    # Use StyleGAN2 version
    elif args['stylegan_ver'] == "2":
        if args['category'] == "eyes_256":
            print("Eyes category")
            resolution = 256
            max_layer = 7 #7 # TODO this max layer stuff yo!
        if args['category'] == "eyes_512":
            print("Eyes category")
            resolution = 512
            max_layer = 7
        else:
            assert "Not implementated!"

        print("---- Resolution:", resolution, " Layers:", max_layer)

        print("---- Get avg latent")
        avg_latent =  np.load(args['exp_dir'] + "/" + args['average_latent'])
        print("---- Latent to torch")
        avg_latent = torch.from_numpy(avg_latent).type(torch.FloatTensor).to(device)
        # avg_latent = torch.ones((14, 512)).type(torch.FloatTensor).to(device)
        
        print("AVG latent", avg_latent.shape)
        
        print("----  Build Generator")

        gpus = 1

        path_to_pretrained = args['stylegan_checkpoint']
        print(f'Resuming from "{path_to_pretrained}"')
        
        print("Prepare StytleGAN2")
        ##########################################
        g_all, _, _ = prepare_SG2(resolution, path_to_pretrained, avg_latent, max_layer, gpus, device, save_intermediate_results=True)
        
        print("======") 

        g_all.eval()
        
        print("---- Parallel")
        g_all = nn.DataParallel(g_all, device_ids=device_ids).to(device)#.cuda()


    else:
        assert "Not implementated error"


    print("---- Create Upsamplers")
    res  = args['dim'][1]
    mode = args['upsample_mode']
    upsamplers = [nn.Upsample(scale_factor=res / 4, mode=mode, align_corners=False),
                  nn.Upsample(scale_factor=res / 4, mode=mode, align_corners=False),
                  nn.Upsample(scale_factor=res / 8, mode=mode, align_corners=False),
                  nn.Upsample(scale_factor=res / 8, mode=mode, align_corners=False),
                  nn.Upsample(scale_factor=res / 16, mode=mode, align_corners=False),
                  nn.Upsample(scale_factor=res / 16, mode=mode, align_corners=False),
                  nn.Upsample(scale_factor=res / 32, mode=mode, align_corners=False),
                  nn.Upsample(scale_factor=res / 32, mode=mode, align_corners=False),
                  nn.Upsample(scale_factor=res / 64, mode=mode, align_corners=False),
                  nn.Upsample(scale_factor=res / 64, mode=mode, align_corners=False),
                  nn.Upsample(scale_factor=res / 128, mode=mode, align_corners=False),
                  nn.Upsample(scale_factor=res / 128, mode=mode, align_corners=False),
                  nn.Upsample(scale_factor=res / 256, mode=mode, align_corners=False),
                  nn.Upsample(scale_factor=res / 256, mode=mode, align_corners=False)
                  ]

    if resolution > 256:
        upsamplers.append(nn.Upsample(scale_factor=res / 512, mode=mode, align_corners=False))
        upsamplers.append(nn.Upsample(scale_factor=res / 512, mode=mode, align_corners=False))

    if resolution > 512:

        upsamplers.append(Interpolate(res, 'bilinear'))
        upsamplers.append(Interpolate(res, 'bilinear'))

    print("---- Done")



    return g_all, avg_latent, upsamplers


def generate_data(args, checkpoint_path, num_sample, start_step=0, vis=True, ignore_latent_layers=None):
    if args['category'] == 'car':
        from torch_utils.data_util import car_20_palette as palette
    elif args['category'] == 'face':
        from torch_utils.data_util import face_palette as palette
    elif args['category'] == 'bedroom':
        from torch_utils.data_util import bedroom_palette as palette
    elif args['category'] == 'cat':
        from torch_utils.data_util import cat_palette as palette
    elif "eyes" in args['category']:
        from torch_utils.data_util import cat_palette as palette

    else:
        assert False
    if not vis:
        result_path = os.path.join(checkpoint_path, 'samples_' + str(num_sample) )
    else:
        result_path = os.path.join(checkpoint_path, 'vis_%d'%num_sample)
    if os.path.exists(result_path):
        pass
    else:
        os.system('mkdir -p %s' % (result_path))
        print('Experiment folder created at: %s' % (result_path))


    classifier_path = os.path.join(checkpoint_path, "models")
    g_all, avg_latent, upsamplers = prepare_stylegan(args)

    classifier_list = []
    for MODEL_NUMBER in range(args['model_num']):
        print('MODEL_NUMBER', MODEL_NUMBER)

        classifier = pixel_classifier(numpy_class=args['number_class']
                                      , dim=args['dim'][-1])
        classifier =  nn.DataParallel(classifier, device_ids=device_ids).to(device)#.cuda()

        #print(classifier)
        path_to_model = os.path.join(classifier_path, 'model_' + str(MODEL_NUMBER) + '.pth')
        
        print(path_to_model)
        checkpoint = torch.load(path_to_model)
        #print(checkpoint)
        classifier.load_state_dict(checkpoint['model_state_dict'])


        classifier.eval()
        classifier_list.append(classifier)


    exec_times = []

    softmax_f = nn.Softmax(dim=1)
    with torch.no_grad():
        latent_cache = []
        image_cache, image_NIR_cache = [], []
        seg_cache = []
        entropy_calculate = []
        results = []
        # np.random.seed(start_step)
        count_step = start_step



        print( "num_sample: ", num_sample)


        for i in tqdm(range(num_sample)):
            #if i % 100 == 0:
            #    print("Generate", i, "Out of:", num_sample)

            curr_result = {}

            latent = np.random.randn(1, 512)

            curr_result['latent'] = latent


            latent = torch.from_numpy(latent).type(torch.FloatTensor).to(device)
            latent_cache.append(latent)

            img, img_NIR, affine_layers = latent_to_image(g_all, upsamplers, latent, dim=args['dim'][1],
                                                     return_upsampled_layers=True, 
                                                     ignore_latent_layers=ignore_latent_layers, dev=device)

            if args['dim'][0] != args['dim'][1]:
                img = img[:, 64:448][0]
                img_NIR = img_NIR[:, 64:448][0]
            else:
                img = img[0]
                img_NIR = img_NIR[0]

            image_cache.append(img)
            image_NIR_cache.append(img_NIR)

            if args['dim'][0] != args['dim'][1]:
                affine_layers = affine_layers[:, :, 64:448]
            affine_layers = affine_layers[0]

            print("Affine layers from latent_to_image:", affine_layers.shape)
            affine_layers = affine_layers.reshape(args['dim'][-1], -1).transpose(1, 0)
            print("Affine layers post reshape, transpose:", affine_layers.shape)
            all_seg = []
            all_entropy = []
            mean_seg = None

            start_time = time.time()

            seg_mode_ensemble = []
            for MODEL_NUMBER in range(args['model_num']):
                classifier = classifier_list[MODEL_NUMBER]


                #print("AFFINE LAYER SHAPE:", affine_layers.shape)
                #exit()
                img_seg = classifier(affine_layers)
                img_seg = img_seg.squeeze()


                entropy = Categorical(logits=img_seg).entropy()
                all_entropy.append(entropy)

                all_seg.append(img_seg)

                #print(img_seg)
                #print(img_seg.shape)
                #exit()
                if mean_seg is None:
                    mean_seg = softmax_f(img_seg)
                else:
                    mean_seg += softmax_f(img_seg)

                img_seg_final = oht_to_scalar(img_seg)
                img_seg_final = img_seg_final.reshape(args['dim'][0], args['dim'][1], 1)
                
                img_seg_final = img_seg_final.cpu().detach().numpy()

                seg_mode_ensemble.append(img_seg_final)

            mean_seg = mean_seg / len(all_seg)

            full_entropy = Categorical(mean_seg).entropy()

            js = full_entropy - torch.mean(torch.stack(all_entropy), 0)

            top_k = js.sort()[0][- int(js.shape[0] / 10):].mean()
            entropy_calculate.append(top_k)

            cur_time = time.time()
            exec_times.append(cur_time - start_time)
            start_time = cur_time

            img_seg_final = np.concatenate(seg_mode_ensemble, axis=-1)
            img_seg_final = scipy.stats.mode(img_seg_final, 2)[0].reshape(args['dim'][0], args['dim'][1])
            

            

            del (affine_layers)
            if vis:

                color_mask = 0.7 * colorize_mask(img_seg_final, palette) + 0.3 * img

                #scipy.misc.imsave(os.path.join(result_path, "vis_" + str(i) + '.jpg'), color_mask.astype(np.uint8))
                imageio.imwrite(os.path.join(result_path, "vis_" + str(i) + '.jpg'), color_mask.astype(np.uint8))

                #scipy.misc.imsave(os.path.join(result_path, "vis_" + str(i) + '_image.jpg'), img.astype(np.uint8))
                imageio.imwrite(os.path.join(result_path, "vis_" + str(i) + '_image.jpg'), img.astype(np.uint8))
                

            else:
                seg_cache.append(img_seg_final)
                curr_result['uncertrainty_score'] = top_k.item()
                image_label_name = os.path.join(result_path, 'image_' + str(count_step) + '_label.png')
                image_name = os.path.join(result_path, str(count_step) + '_RGB.png')
                image_NIR_name = os.path.join(result_path,  str(count_step) + '_NIR.png')

                js_name = os.path.join(result_path, str(count_step) + '.npy')
                img = Image.fromarray(img)

                img_NIR = img_NIR[:, :, 0]
                img_NIR = Image.fromarray(img_NIR, mode="L")
                
                img_seg = Image.fromarray(img_seg_final.astype('uint8'))
                
                js = js.cpu().numpy().reshape(args['dim'][0], args['dim'][1])
                
                
                img.save(image_name)
                img_NIR.save(image_NIR_name)
                img_seg.save(image_label_name)
                
                # SAVE THE NPY file? too large
                #np.save(js_name, js)
                curr_result['image_name'] = image_name
                curr_result['image_label_name'] = image_label_name
                curr_result['js_name'] = js_name
                count_step += 1


                results.append(curr_result)
                if i % 1000 == 0 and i != 0:
                    with open(os.path.join(result_path, str(i) + "_" + str(start_step) + '.pickle'), 'wb') as f:
                        pickle.dump(results, f)


        #print(result_path)
        #np.save(result_path + "_test_times", exec_times)

        print("Run times:")
        print( np.round(np.mean(exec_times) * 1000, 2) , "\pm", np.round(np.std(exec_times) * 1000, 2)  )
        
        with open(os.path.join(result_path, str(num_sample) + "_" + str(start_step) + '.pickle'), 'wb') as f:
            pickle.dump(results, f)
    

def prepare_data(args, palette, ignore_latent_layers=None):
    print("-- Prepare stylegan")
    g_all, avg_latent, upsamplers = prepare_stylegan(args)
    print("-- Get latent info")
    latent_all = np.load(args['exp_dir'] + "/" + args['annotation_image_latent_path'])
    latent_all = torch.from_numpy(latent_all).to(device)#.cuda()
    
    print("latent all:", latent_all.shape)

    #input_already_w = args['input_already_w']

    print("-- load mask")
    # load annotated mask
    mask_list = []
    im_list, im_NIR_list = [], []
    latent_all = latent_all[:args['max_training']]
    num_data = len(latent_all)

    print("==" * 30)
    print("Go over latents")
    for i in range(len(latent_all)):
        
        if i >= args['max_training']:
            break
        name = 'image_mask%0d.npy' % i

        im_frame = np.load(os.path.join( args['exp_dir'], args['annotation_mask_path'] , name))
        mask = np.array(im_frame)
        # TODO 
        #mask =  cv2.resize(np.squeeze(mask), dsize=(args['dim'][1], args['dim'][0]), interpolation=cv2.INTER_NEAREST) 
        mask = Image.fromarray(np.squeeze(mask))
        mask = mask.resize((args['dim'][1], args['dim'][0]), Image.NEAREST)
        #mask =  cv2.resize(np.squeeze(mask), , interpolation=cv2.INTER_NEAREST)
        mask = np.array(mask)
        mask_list.append(mask)

        im_name = os.path.join( args['exp_dir'], args['annotation_mask_path'], 'image_%d.jpg' % i)
        im_NIR_name = os.path.join( args['exp_dir'], args['annotation_mask_path'], 'image_%d_NIR.jpg' % i)
        
        img = Image.open(im_name)
        img = img.resize((args['dim'][1], args['dim'][0]))

        img_NIR = Image.open(im_NIR_name).convert("RGB")
        img_NIR = img_NIR.resize((args['dim'][1], args['dim'][0]))

        im_list.append(np.array(img))
        im_NIR_list.append(np.array(img_NIR))


    print("-- clean up masks")
    # delete small annotation error
    for i in range(len(mask_list)):  # clean up artifacts in the annotation, must do
        for target in range(1, 50):
            if (mask_list[i] == target).sum() < 30:
                mask_list[i][mask_list[i] == target] = 0


    all_mask = np.stack(mask_list)

    print("-- Generate all training data for pixel classifer")
    # 3. Generate ALL training data for training pixel classifier


    # so all features_maps == 256 * 256 * 64 ? ,  4992
    print("Len latent all:", len(latent_all) )
    all_feature_maps_train = np.zeros((args['dim'][0] * args['dim'][1] * len(latent_all), args['dim'][2]), dtype=np.float16)
    all_mask_train = np.zeros((args['dim'][0] * args['dim'][1] * len(latent_all),), dtype=np.float16)

    print("all_feature_maps TRAIN: ", all_feature_maps_train.shape)

    print("Show training examples")
    vis, vis_NIR = [], []
    for i in range(len(latent_all) ):

        gc.collect()

        latent_input = latent_all[i].float()
        
        print("Latent input size:", latent_input.shape, latent_input.unsqueeze(0).shape)

        # TODO fmain difference here is that this uses retun_upsampled = True and use_style_latents ... 
        # while make training_data script uses different
        img, img_NIR, feature_maps = latent_to_image(g_all, upsamplers, latent_input.unsqueeze(0), dim=args['dim'][1],
                                            return_upsampled_layers=True, use_style_latents=args['annotation_data_from_w'],
                                            ignore_latent_layers = ignore_latent_layers, dev=device) # TODO ignore layers how many

        print("Feature maps from (latent to image):", feature_maps.shape )
        #if args['dim'][0]  != args['dim'][1]:
        #    print("DO THIS OR NOT?")
            # only for car
        #    img = img[:, 64:448]
        #    feature_maps = feature_maps[:, :, 64:448]
        
        mask = all_mask[i:i + 1]
        feature_maps = feature_maps.permute(0, 2, 3, 1)

        # Permute: torch.Size([1, 256, 256, 4992])
        print("Permute:", feature_maps.shape)

        feature_maps = feature_maps.reshape(-1, args['dim'][2])
        print("reshape:", feature_maps.shape)
        

        new_mask =  np.squeeze(mask)
        mask = mask.reshape(-1)

        all_feature_maps_train[args['dim'][0] * args['dim'][1] * i: args['dim'][0] * args['dim'][1] * i + args['dim'][0] * args['dim'][1]] = feature_maps.cpu().detach().numpy().astype(np.float16)
        all_mask_train[args['dim'][0] * args['dim'][1] * i:args['dim'][0] * args['dim'][1] * i + args['dim'][0] * args['dim'][1]] = mask.astype(np.float16)


        
        # TODO
        #img_show =  cv2.resize(np.squeeze(img[0]), dsize=(args['dim'][1], args['dim'][1]), interpolation=cv2.INTER_NEAREST)
        #print("IMG SHAPE:", img[0].shape)
        img_show = Image.fromarray(np.squeeze(img[0]))
        #print("IMG SHAPE:", np.array(img_show).shape)
        img_show = img_show.resize( (args['dim'][1], args['dim'][1]), Image.NEAREST)

        
        img_NIR_show = Image.fromarray(np.squeeze(img_NIR[0])).convert("RGB")
        img_NIR_show = img_NIR_show.resize( (args['dim'][1], args['dim'][1]), Image.NEAREST)

        print(img_show)
        print(img_NIR_show)
        
        
        #print("--- After resize:", np.array(img_show).shape)
        curr_vis = np.concatenate( [im_list[i], img_show, get_image_difference(im_list[i], img_show), im_NIR_list[i], img_NIR_show, colorize_mask(new_mask, palette)], 0 )
        
        vis.append( curr_vis )

    vis = np.concatenate(vis, 1)
    #scipy.misc.imsave(os.path.join(args['exp_dir'], "train_data.jpg"), vis)
    # TODO this is wrong
    imageio.imwrite(os.path.join(args['exp_dir'], "train_data.jpg"), vis)


    print("FINAL all_feature_maps_train", all_feature_maps_train.shape)
    return all_feature_maps_train, all_mask_train, num_data


def main(args, ignore_latent_layers=None):

    torch.manual_seed(0) # NOTE keep this the same as the seed in make_training_data

    if args['category'] == 'car':
        from torch_utils.data_util import car_20_palette as palette
    elif args['category'] == 'face':
        from torch_utils.data_util import face_palette as palette
    elif args['category'] == 'bedroom':
        from torch_utils.data_util import bedroom_palette as palette
    elif args['category'] == 'cat':
        from torch_utils.data_util import cat_palette as palette
    elif "eyes" in args['category']:
        from torch_utils.data_util import cat_palette as palette

    

    print("Prepare data")
    all_feature_maps_train_all, all_mask_train_all, num_data = prepare_data(args, palette, ignore_latent_layers=ignore_latent_layers)
    print("-- Done")
    
    train_data = trainData(torch.FloatTensor(all_feature_maps_train_all),
                           torch.FloatTensor(all_mask_train_all))

    
    count_dict = get_label_stas(train_data)
    
    max_label = max([*count_dict])
    print(" *********************** max_label " + str(max_label) + " ***********************")


    print(" *********************** Current number data " + str(num_data) + " ***********************")


    batch_size = args['batch_size']

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    print(" *********************** Current dataloader length " +  str(len(train_loader)) + " ***********************")

    os.makedirs(os.path.join(args['exp_dir'], 'models'))
    for MODEL_NUMBER in range(args['model_num']):

        gc.collect()

        classifier = pixel_classifier(numpy_class=(max_label + 1), dim=args['dim'][-1])

        classifier.init_weights()

        classifier = nn.DataParallel(classifier, device_ids=device_ids).to(device)#.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=0.001)
        classifier.train()


        iteration = 0
        break_count = 0
        best_loss = 10000000
        stop_sign = 0
        for epoch in tqdm(range(100)):
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_batch = y_batch.type(torch.long)
                y_batch = y_batch.type(torch.long)

                #print("X BATCH:", X_batch.shape)
                # AFFINE LAYER SHAPE: torch.Size([64, 4992])

                optimizer.zero_grad()
                y_pred = classifier(X_batch)
                #print("PRED:", y_pred.shape)
                #print("Y_true:", y_batch.shape)
                loss = criterion(y_pred, y_batch)
                acc = multi_acc(y_pred, y_batch)

                loss.backward()
                optimizer.step()

                #exit()
                iteration += 1
                if iteration % 1000 == 0:
                    print('Epoch : ', str(epoch), 'iteration', iteration, 'loss', loss.item(), 'acc', acc)
                    gc.collect()


                if iteration % 5000 == 0:
                    model_path = os.path.join(args['exp_dir'], 'models',
                                              'model_20parts_iter' +  str(iteration) + '_number_' + str(MODEL_NUMBER) + '.pth')
                    print('Save checkpoint, Epoch : ', str(epoch), ' Path: ', model_path)

                    torch.save({'model_state_dict': classifier.state_dict()},
                               model_path)

                if epoch > 3:
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        break_count = 0
                    else:
                        break_count += 1

                    if break_count > 50:
                        stop_sign = 1
                        print("*************** Break, Total iters,", iteration, ", at epoch", str(epoch), "***************")
                        break

            if stop_sign == 1:
                break

        gc.collect()
        model_path = os.path.join(args['exp_dir'], 'models', 'model_' + str(MODEL_NUMBER) + '.pth')
        MODEL_NUMBER += 1
        print('save to:',model_path)
        torch.save({'model_state_dict': classifier.state_dict()},
                   model_path)
        gc.collect()


        gc.collect()
        torch.cuda.empty_cache()    # clear cache memory on GPU


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str)
    parser.add_argument('--exp_dir', type=str,  default="")
    parser.add_argument('--generate_data', type=bool, default=False)
    parser.add_argument('--save_vis', type=bool, default=False)
    parser.add_argument('--start_step', type=int, default=0)

    parser.add_argument('--resume', type=str,  default="")
    parser.add_argument('--num_sample', type=int,  default=1000)

    parser.add_argument('--upsample_threshold', type=int,  default=-1)
    #parser.add_argument('--input_already_w', type=bool, default=False) # already done as args["annotation_data_from_w"]

    ignore_latent_layers = None # could be 4 or 8 ... how many latent layers to ignore (use less layers for uspampling)

    args = parser.parse_args()

    opts = json.load(open(args.exp, 'r'))
    print("Opt", opts)

    if args.exp_dir != "":
        opts['exp_dir'] = args.exp_dir


    path =opts['exp_dir']
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path))

    os.system('cp %s %s' % (args.exp, opts['exp_dir']))



    if args.generate_data:
        print("Generate data")
        generate_data(opts, args.resume, args.num_sample, vis=args.save_vis, start_step=args.start_step, ignore_latent_layers=ignore_latent_layers)
    else:

        main(opts, ignore_latent_layers=ignore_latent_layers)

