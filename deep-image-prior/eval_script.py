import os

# for num_layers in [5,7,3]:
#     for num_heads in [8,6,12]:
#         for patch_size in [32, 16]:
#             for emb_dim in [192,96]:
#                 os.system(f'python image_prior.py denoise transformer --num_layers {num_layers} --num_heads {num_heads} --emb_dim {emb_dim} --patch_size {patch_size}  --image_path data/denoising/snail.jpg --iters 5000')


# for num_layers in [8, 10, 12, 14]:
#     for num_channels in [8, 10, 12, 16]:
#         os.system(f'python image_prior.py denoise resnet --num_layers {num_layers} --num_channels {num_channels}  --image_path data/denoising/F16_GT.png --iters 5000')

# os.system(f'python image_prior.py denoise orig --image_path data/denoising/F16_GT.png --iters 5000')


# for num_layers in [5,7,3]:
#     for num_heads in [8,6,12]:
#         for patch_size in [32, 16]:
#             for emb_dim in [192,96]:
#                 os.system(f'python image_prior.py inpaint transformer --num_layers {num_layers} --num_heads {num_heads} --emb_dim {emb_dim} --patch_size {patch_size}  --image_path data/inpainting/kate.png --mask_path data/inpainting/kate_mask.png --iters 7000')


# for num_layers in [8, 10, 12, 14]:
#     for num_channels in [8, 10, 12, 16]:
#         os.system(f'python image_prior.py inpaint resnet --num_layers {num_layers} --num_channels {num_channels}  --image_path data/inpainting/kate.png --mask_path data/inpainting/kate_mask.png --iters 7000')

os.system(f'python image_prior.py denoise orig --num_layers 5 --num_heads 8 --emb_dim 192 --patch_size 32  --image_path data/denoising/F16_GT.png --iters 5000')

os.system(f'python image_prior.py denoise transformer --num_layers 5 --num_heads 8 --emb_dim 192 --patch_size 32  --image_path data/denoising/F16_GT.png --iters 5000')

os.system(f'python image_prior.py denoise swin_transformer --num_layers 5 --num_heads 8 --emb_dim 192 --patch_size 32  --image_path data/denoising/F16_GT.png --iters 5000')

os.system(f'python image_prior.py denoise transformercnn3 --num_layers 5 --num_heads 8 --emb_dim 192 --patch_size 32  --image_path data/denoising/F16_GT.png --iters 5000')

os.system(f'python image_prior.py denoise transformercnn3before --num_layers 5 --num_heads 8 --emb_dim 192 --patch_size 32  --image_path data/denoising/F16_GT.png --iters 5000')


# os.system(f'python image_prior.py inpaint orig --image_path data/inpainting/kate.png --mask_path data/inpainting/kate_mask.png --iters 7000')

# os.system(f'python image_prior.py inpaint orig --image_path data/inpainting/vase.png --mask_path data/inpainting/vase_mask.png --iters 7000')


# os.system(f'python image_prior.py inpaint transformercnn3before --num_layers 5 --num_heads 8 --emb_dim 192 --patch_size 32 --image_path data/inpainting/kate.png --mask_path data/inpainting/kate_mask.png --iters 10000')

# os.system(f'python image_prior.py inpaint transformercnn3 --num_layers 5 --num_heads 8 --emb_dim 192 --patch_size 32 --image_path data/inpainting/kate.png --mask_path data/inpainting/kate_mask.png --iters 10000')

# os.system(f'python image_prior.py inpaint transformer --num_layers 5 --num_heads 8 --emb_dim 192 --patch_size 32 --image_path data/inpainting/kate.png --mask_path data/inpainting/kate_mask.png --iters 7000')

# os.system(f'python image_prior.py inpaint transformercnn3before --num_layers 5 --num_heads 8 --emb_dim 192 --patch_size 32 --image_path data/inpainting/vase.png --mask_path data/inpainting/vase_mask.png --iters 10000')

# os.system(f'python image_prior.py inpaint transformercnn3 --num_layers 5 --num_heads 8 --emb_dim 192 --patch_size 32 --image_path data/inpainting/vase.png --mask_path data/inpainting/vase_mask.png --iters 10000')

# os.system(f'python image_prior.py inpaint transformer --num_layers 5 --num_heads 8 --emb_dim 192 --patch_size 32 --image_path data/inpainting/vase.png --mask_path data/inpainting/vase_mask.png --iters 7000')


# os.system(f'python image_prior.py inpaint swin_transformer --num_layers 5 --num_heads 8 --emb_dim 192 --patch_size 32 --image_path data/inpainting/kate.png --mask_path data/inpainting/kate_mask.png --iters 7000')



