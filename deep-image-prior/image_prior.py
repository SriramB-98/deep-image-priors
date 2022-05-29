from __future__ import print_function
import matplotlib.pyplot as plt

import os
import argparse

import numpy as np
from models import *
from models.transformer import *
import torch
import torch.optim

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.denoising_utils import *
from utils.inpainting_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model(args, image_size, input_depth):
    
    if args.model == 'transformer':
        net = Transformer(  image_size=image_size,
                            patch_size=args.patch_size,
                            num_layers=args.num_layers,
                            num_heads=args.num_heads,
                            hidden_dim=args.emb_dim,
                            mlp_dim=args.emb_dim).to(device)
        hyperparams = f'{args.patch_size}_{args.emb_dim}_{args.num_layers}_{args.num_heads}'
    elif args.model == 'transformercnn':
        net = TransformerWithCNN(  image_size=image_size,
                            patch_size=args.patch_size,
                            num_layers=args.num_layers,
                            num_heads=args.num_heads,
                            hidden_dim=args.emb_dim,
                            mlp_dim=args.emb_dim).to(device)
        hyperparams = f'{args.patch_size}_{args.emb_dim}_{args.num_layers}_{args.num_heads}'
    elif args.model == 'transformercnn3':
        net = TransformerWithCNN(3, image_size=image_size,
                            patch_size=args.patch_size,
                            num_layers=args.num_layers,
                            num_heads=args.num_heads,
                            hidden_dim=args.emb_dim,
                            mlp_dim=args.emb_dim, ).to(device)
        hyperparams = f'{args.patch_size}_{args.emb_dim}_{args.num_layers}_{args.num_heads}'
    elif args.model == 'transformercnn3before':
        net = TransformerWithCNN(-3, image_size=image_size,
                            patch_size=args.patch_size,
                            num_layers=args.num_layers,
                            num_heads=args.num_heads,
                            hidden_dim=args.emb_dim,
                            mlp_dim=args.emb_dim).to(device)
        hyperparams = f'{args.patch_size}_{args.emb_dim}_{args.num_layers}_{args.num_heads}'
    elif args.model == 'swin_transformer':
        net = SwinTransformer(image_size=512,
                        patch_size=4,
                        num_layers=5,
                        num_heads=4,
                        hidden_dim=192,
                        mlp_ratio=1).to(device)
#         net = SwinTransformer(image_size=image_size,
#                             patch_size=8,
#                             num_layers=args.num_layers,
#                             num_heads=args.num_heads,
#                             hidden_dim=args.emb_dim).to(device)
        hyperparams = f'8_{args.emb_dim}_{args.num_layers}_{args.num_heads}'
    elif args.model == 'resnet':
        net =  ResNet(input_depth, 3, args.num_layers, args.num_channels).to(device)
        hyperparams = f'{args.num_layers}_{args.num_channels}'
    elif args.model == 'orig':
        if args.task == 'denoise':
            net = get_net(input_depth, 'skip', 'reflection',
                  skip_n33d=128, 
                  skip_n33u=128, 
                  skip_n11=4, 
                  num_scales=5,
                  upsample_mode='bilinear').type(dtype)
        elif args.task == 'inpaint':
            net = skip(input_depth, 3, 
                       num_channels_down = [128] * 5,
                       num_channels_up =   [128] * 5,
                       num_channels_skip =    [128] * 5,  
                       filter_size_up = 3, filter_size_down = 3, 
                       upsample_mode='nearest', filter_skip_size=1,
                       need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU').type(dtype)
        hyperparams = 'default'
    return net, hyperparams
    
def inpaint_task(args):
    PLOT = True
    imsize = -1
    dim_div_by = 64
    show_every = 50
    LR = 0.0001
    OPT_OVER = 'net'
    OPTIMIZER = 'adam'
    img_path = args.image_path
    mask_path = args.mask_path
    img_pil, img_np = get_image(img_path, imsize)
    image_size = img_np.shape[-1]
    print(image_size)
    img_mask_pil, img_mask_np = get_image(mask_path, imsize)
    img_mask_pil = crop_image(img_mask_pil, dim_div_by)
    img_pil      = crop_image(img_pil,      dim_div_by)
    img_np      = pil_to_np(img_pil)
    img_mask_np = pil_to_np(img_mask_pil)
    img_mask_var = np_to_torch(img_mask_np).type(dtype)
    num_iter = args.iters
    if 'vase.png' in img_path:
        INPUT = 'noise'
        input_depth = 3
        param_noise = False
        figsize = 5
        reg_noise_std = 0.03
    elif ('kate.png' in img_path) or ('peppers.png' in img_path):
        INPUT = 'noise'
        input_depth = 3
        param_noise = False
        figsize = 5
        reg_noise_std = 0.01
    elif 'library.png' in img_path:
        INPUT = 'noise'
        input_depth = 3
        figsize = 8
        reg_noise_std = 0.00
        param_noise = True
    else:
        assert False
    
    net, hyperparams = get_model(args, image_size, input_depth)
    net = net.type(dtype)
    net_input = get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype)
    mse = torch.nn.MSELoss().type(dtype)
    img_var = np_to_torch(img_np).type(dtype)
    mask_var = np_to_torch(img_mask_np).type(dtype)
    
    image_name = img_path.split('/')[-1].split('.')[0]
    fpath = os.path.join(args.out_dir, args.task, image_name, args.model, hyperparams) 
    images_path = os.path.join(fpath, 'images')
    try:
        os.makedirs(images_path)
    except Exception as e:
        print(e)
    lossplot_path = os.path.join(fpath, 'loss.png')
    log_path = os.path.join(fpath, 'logs.txt')
    log_fp = open(log_path, 'a')
    log_fp.write('Iteration, Loss\n')
    
    i = 0
    loss_list = []
    glob_var_list = [i]
    def closure(var_list):

        i = var_list[0]

        if param_noise:
            for n in [x for x in net.parameters() if len(x.size()) == 4]:
                n = n + n.detach().clone().normal_() * n.std() / 50

        net_input = net_input_saved
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)


        out = net(net_input)

        total_loss = mse(out * mask_var, img_var * mask_var)
        total_loss.backward()
        test_loss = mse(out, img_var).detach().item()
        print ('Iteration %05d    Loss %f, %f' % (i, total_loss.item(), test_loss), '\r', end='')
        log_fp.write('%05d, %f, %f \n' % (i, total_loss.item(), test_loss) )
        loss_list.append(test_loss)#total_loss.item())
        
        if  PLOT and i % show_every == 0:
            out_np = torch_to_np(out)
            plot_image_grid([np.clip(out_np, 0, 1)], save_path=os.path.join(images_path, f'{i}.png'), factor=figsize, nrow=1)

        var_list[0] += 1

        return total_loss

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    p = get_params(OPT_OVER, net, net_input)
    optimize(OPTIMIZER, p, closure, LR, num_iter, glob_var_list)
    
    log_fp.close()
#     out_np = torch_to_np(net(net_input))
#     q = plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13);
    start = 100
    plt.plot(np.arange(start, len(loss_list)), loss_list[start:], label='MSE loss')
    plt.xlabel('iteration')
    plt.ylabel('mean squared error')
    plt.legend()
    plt.savefig(lossplot_path)
    plt.close()
    
    return
    
def denoise_task(args):
    imsize =-1
    PLOT = True
    sigma = 25
    sigma_ = sigma/255
    INPUT = 'noise' # 'meshgrid'
    pad = 'reflection'
    OPT_OVER = 'net' # 'net,input'
    reg_noise_std = 0#1./30. # set to 1./20. for sigma=50
    LR = 0.0001
    OPTIMIZER='adam' # 'LBFGS'
    show_every = 100
    exp_weight=0.99
    num_iter = args.iters
    input_depth = 3
    
    fname = args.image_path
    if 'snail.jpg' in fname:
        img_noisy_pil = crop_image(get_image(fname, imsize)[0], d=32)
        img_noisy_np = pil_to_np(img_noisy_pil)
        # As we don't have ground truth
        img_pil = img_noisy_pil
        img_np = img_noisy_np
        figsize = 5
        image_size = img_np.shape[-1]
    elif 'F16_GT.png' in fname:
        # Add synthetic noise
        img_pil = crop_image(get_image(fname, imsize)[0], d=32)
        img_np = pil_to_np(img_pil)
        img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
        figsize = 4
        image_size = img_np.shape[-1]
    else:
        raise Exception('Image not handled')
    
    net, hyperparams = get_model(args, image_size, input_depth)
    
    image_name = fname.split('/')[-1].split('.')[0]
    fpath = os.path.join(args.out_dir, args.task, image_name, args.model, hyperparams) 
    lossplot_path = os.path.join(fpath, 'loss.png')
    psnrplot_path = os.path.join(fpath, 'psnr.png')
    images_path = os.path.join(fpath, 'images')
    try:
        os.makedirs(images_path)
    except Exception as e:
        print(e)
    log_path = os.path.join(fpath, 'logs.txt')
    log_fp = open(log_path, 'a')
    net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()
    s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
    print ('Number of params: %d' % s)
    mse = torch.nn.MSELoss().type(dtype)
    img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)
    
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    out_avg = None
    last_net = None
    loss_list, psrn_noisy_list, psrn_gt_list = [], [], []
    psrn_noisy_last = 0
    i = 0
    log_fp.write('Iteration, Loss, PSNR_noisy, PSNR_gt \n')
    glob_var_list = [i,log_fp, out_avg, psrn_noisy_last, last_net]
    def closure(var_list):
        i,log_fp, out_avg, psrn_noisy_last, last_net = var_list

#         if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input)

        # Smoothing
        if out_avg is None:
            out_avg = var_list[2] = out.detach()
        else:
            out_avg = var_list[2] = out_avg * exp_weight + out.detach() * (1 - exp_weight)

        total_loss = mse(out, img_noisy_torch)
        total_loss.backward()
        psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0]) 
        psrn_gt    = compare_psnr(img_np, out.detach().cpu().numpy()[0]) 
        psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0]) 

        # Note that we do not have GT for the "snail" example
        # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
        print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSNR_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
        log_fp.write('%05d, %f, %f, %f, %f \n' % (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm) )
        
        if i % show_every:
            if psrn_noisy - psrn_noisy_last < -5: 
                print('Falling back to previous checkpoint.')

                for new_param, net_param in zip(last_net, net.parameters()):
                    net_param.data.copy_(new_param.cuda())

                return total_loss*0
            else:
                var_list[4] = [x.detach().cpu() for x in net.parameters()]
                var_list[3] = psrn_noisy
        
        if  PLOT and i % show_every == 0:
            out_np = torch_to_np(out)
            plot_image_grid([np.clip(out_np, 0, 1)], save_path=os.path.join(images_path, f'{i}.png'), factor=figsize, nrow=1)
            
        loss_list.append(total_loss.item())
        psrn_noisy_list.append(psrn_noisy)
        psrn_gt_list.append(psrn_gt)
        var_list[0] += 1
        return total_loss
    

    p = get_params(OPT_OVER, net, net_input)
    optimize(OPTIMIZER, p, closure, LR, num_iter, glob_var_list)
    log_fp.close()
#     out_np = torch_to_np(net(net_input))
#     q = plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13);
    start = 100
    plt.plot(np.arange(start, len(loss_list)), loss_list[start:], label='MSE loss')
    plt.xlabel('iteration')
    plt.ylabel('mean squared error')
    plt.legend()
    plt.savefig(lossplot_path)
    plt.close()
    
    plt.plot(psrn_noisy_list, label='PSNR between output and noisy image')
    plt.plot(psrn_gt_list, label='PSNR between output and ground truth')
    plt.xlabel('iteration')
    plt.ylabel('PSNR')
    plt.legend()
    plt.savefig(psnrplot_path)
    plt.close()
    
    return
    


def get_args():
    parser = argparse.ArgumentParser(description='argument parser')
    parser.add_argument('task', type=str,
                        help='task to evaluate model on')
    parser.add_argument('model', type=str,
                        help='which model')
    parser.add_argument('--iters',  type=int, default=10000,
                        help='Iterations')
    parser.add_argument('--num_layers',  type=int, default=None,
                        help='Number of layers (for transformers)')
    parser.add_argument('--num_heads',  type=int, default=None,
                        help='Number of heads (for transformers)')
    parser.add_argument('--emb_dim',  type=int, default=None,
                        help='Embedding dimensions (for transformers)')
    parser.add_argument('--patch_size',  type=int, default=None,
                        help='Patch size (for transformers)')
    parser.add_argument('--out_dir',  type=str, default='/cmlscratch/sriramb/dipexps/',
                        help='output directory')
    parser.add_argument('--image_path',  type=str, default=None,
                        help='image path')
    parser.add_argument('--mask_path',  type=str, default=None,
                        help='mask path')
    parser.add_argument('--num_channels',  type=int, default=None,
                        help='output directory')
    args = parser.parse_args()
    return args


def Main(args):
    task = args.task
    if task == 'denoise':
        denoise_task(args)
    elif task == 'inpaint':
        inpaint_task(args)
    elif task == 'restore':
        restore_task(args)
    else:
        raise Exception('Task not found')
    return


if __name__ == '__main__':
    args = get_args()
    Main(args)


