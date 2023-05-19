
import dnnlib
from models.stylegan1 import Truncation
import torch
from collections import OrderedDict
from training_scripts_DB_SG2 import legacy
from torch_utils import misc
from torch_utils.utils import  Interpolate

def prepare_SG2_new(network_pkl, device, save_intermediate_results=False):
    #print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda:0')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    print(G)
    # set if we output latent features or not 
    if save_intermediate_results:
        print("Set G.synthesis to save intermediate results")
        G.synthesis.save_intermediate_results = save_intermediate_results

    return G


def prepare_stylegan_and_upsamplers(device, device_ids, args, resolution=256):
    print("---- Resolution:", resolution)

    path_to_pretrained = args['stylegan_checkpoint']
    print(f'Resuming from "{path_to_pretrained}"')
    
    ##########################################
    print("Prepare StytleGAN2")
    #g_all, _, _ = prepare_SG2(resolution, path_to_pretrained, avg_latent, max_layer, gpus, device, save_intermediate_results=True)
    g_all = prepare_SG2_new(path_to_pretrained, device, save_intermediate_results = True)
    g_all.eval()
    
    print("---- Parallel")
    g_all = torch.nn.DataParallel(g_all, device_ids=device_ids).to(device)#.cuda()

    print("---- Create Upsamplers")
    res  = args['dim'][1]
    mode = args['upsample_mode']
    upsamplers = [torch.nn.Upsample(scale_factor=res / 4, mode=mode, align_corners=False),
                  torch.nn.Upsample(scale_factor=res / 4, mode=mode, align_corners=False),
                  torch.nn.Upsample(scale_factor=res / 8, mode=mode, align_corners=False),
                  torch.nn.Upsample(scale_factor=res / 8, mode=mode, align_corners=False),
                  torch.nn.Upsample(scale_factor=res / 16, mode=mode, align_corners=False),
                  torch.nn.Upsample(scale_factor=res / 16, mode=mode, align_corners=False),
                  torch.nn.Upsample(scale_factor=res / 32, mode=mode, align_corners=False),
                  torch.nn.Upsample(scale_factor=res / 32, mode=mode, align_corners=False),
                  torch.nn.Upsample(scale_factor=res / 64, mode=mode, align_corners=False),
                  torch.nn.Upsample(scale_factor=res / 64, mode=mode, align_corners=False),
                  torch.nn.Upsample(scale_factor=res / 128, mode=mode, align_corners=False),
                  torch.nn.Upsample(scale_factor=res / 128, mode=mode, align_corners=False),
                  torch.nn.Upsample(scale_factor=res / 256, mode=mode, align_corners=False),
                  torch.nn.Upsample(scale_factor=res / 256, mode=mode, align_corners=False)
                  ]

    if resolution > 256:
        upsamplers.append(torch.nn.Upsample(scale_factor=res / 512, mode=mode, align_corners=False))
        upsamplers.append(torch.nn.Upsample(scale_factor=res / 512, mode=mode, align_corners=False))

    if resolution > 512:

        upsamplers.append(Interpolate(res, 'bilinear'))
        upsamplers.append(Interpolate(res, 'bilinear'))

    print("---- Done")
    
    return g_all, upsamplers