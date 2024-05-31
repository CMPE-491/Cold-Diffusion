from helpers.resnet_classifier import ResNetClassifier
import torch
from torch import nn
import torch.nn.functional as F
import torch.linalg
import torchvision.transforms as transforms
from torch.utils import data

import numpy as np
from PIL import Image

from .color_utils import rgb2lab, lab2rgb
from .torch_geometry_v3 import get_gaussian_kernel2d,get_gaussian_kernel
from .utils import cycle_with_label

from scipy.ndimage import zoom as scizoom
from kornia.color.gray import rgb_to_grayscale


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]


class ForwardProcessBase:
    
    def forward(self, x,grad, i):
        pass

    @torch.no_grad()
    def reset_parameters(self, batch_size=32):
        pass

class Snow(ForwardProcessBase):
    
    def __init__(self,
                 image_size=(32,32),
                 snow_level=1,
                 num_timesteps=50,
                 snow_base_path=None,
                 random_snow=False,
                 single_snow=False,
                 batch_size=32,
                 load_snow_base=False,
                 fix_brightness=False):
        
        self.num_timesteps = num_timesteps
        self.random_snow = random_snow
        self.snow_level = snow_level
        self.image_size = image_size
        self.single_snow = single_snow
        self.batch_size = batch_size
        self.generate_snow_layer()
        self.fix_brightness = fix_brightness
    
    @torch.no_grad()
    def reset_parameters(self, batch_size=-1):
        if batch_size != -1:
            self.batch_size = batch_size
        if self.random_snow:
            self.generate_snow_layer()



    @torch.no_grad()
    def generate_snow_layer(self):
        if not self.random_snow:
            rstate = np.random.get_state()
            np.random.seed(123321)
        # c[0]/c[1]: mean/std of Gaussian for snowy pixels
        # c[2]: zoom factor
        # c[3]: threshold for snowy pixels
        # c[4]/c[5]: radius/sigma for motion blur
        # c[6]: brightness coefficient
        if self.snow_level == 1:
            c = (0.1, 0.3, 3, 0.5, 5, 4, 0.8)
            snow_thres_start = 0.7
            snow_thres_end = 0.3
            mb_sigma_start = 0.5
            mb_sigma_end = 5.0
            br_coef_start = 0.95
            br_coef_end = 0.7
        elif self.snow_level == 2:
            c = (0.55, 0.3, 2.5, 0.85, 11, 12, 0.55) 
            snow_thres_start = 1.15
            snow_thres_end = 0.7
            mb_sigma_start = 0.05
            mb_sigma_end = 12
            br_coef_start = 0.95
            br_coef_end = 0.55
        elif self.snow_level == 3:
            c = (0.55, 0.3, 2.5, 0.7, 11, 16, 0.4) 
            snow_thres_start = 1.15
            snow_thres_end = 0.7
            mb_sigma_start = 0.05
            mb_sigma_end = 16
            br_coef_start = 0.95
            br_coef_end = 0.4
        elif self.snow_level == 4:
            c = (0.55, 0.3, 2.5, 0.55, 11, 20, 0.3) 
            snow_thres_start = 1.15
            snow_thres_end = 0.55
            mb_sigma_start = 0.05
            mb_sigma_end = 20
            br_coef_start = 0.95
            br_coef_end = 0.3



        self.snow_thres_list = torch.linspace(snow_thres_start, snow_thres_end, self.num_timesteps).tolist()

        self.mb_sigma_list = torch.linspace(mb_sigma_start, mb_sigma_end, self.num_timesteps).tolist()

        self.br_coef_list = torch.linspace(br_coef_start, br_coef_end, self.num_timesteps).tolist()


        self.snow = []
        self.snow_rot = []
        
        if self.single_snow:
            sb_list = []
            for _ in range(self.batch_size):
                cs = np.random.normal(size=self.image_size, loc=c[0], scale=c[1])
                cs = cs[..., np.newaxis]
                cs = clipped_zoom(cs, c[2])
                sb_list.append(cs)
            snow_layer_base = np.concatenate(sb_list, axis=2)
        else:
            snow_layer_base = np.random.normal(size=self.image_size, loc=c[0], scale=c[1])
            snow_layer_base = snow_layer_base[..., np.newaxis]
            snow_layer_base = clipped_zoom(snow_layer_base, c[2])
        
        vertical_snow = False
        if np.random.uniform() > 0.5:
            vertical_snow = True

        for i in range(self.num_timesteps):

            snow_layer = torch.Tensor(snow_layer_base).clone()
            snow_layer[snow_layer < self.snow_thres_list[i]] = 0
            snow_layer = torch.clip(snow_layer, 0, 1)
            snow_layer = snow_layer.permute((2, 0, 1)).unsqueeze(1)
            # Apply motion blur
            kernel_param = get_gaussian_kernel(c[4], self.mb_sigma_list[i])
            motion_kernel = torch.zeros((c[4], c[4]))
            motion_kernel[int(c[4] / 2)] = kernel_param

            horizontal_kernel = motion_kernel[None, None, :]
            horizontal_kernel = horizontal_kernel.repeat(3, 1, 1, 1)
            vertical_kernel = torch.rot90(motion_kernel, k=1, dims=[0,1])
            vertical_kernel = vertical_kernel[None, None, :]
            vertical_kernel = vertical_kernel.repeat(3, 1, 1, 1)

            vsnow = F.conv2d(snow_layer, vertical_kernel, padding='same', groups=1)
            hsnow = F.conv2d(snow_layer, horizontal_kernel, padding='same', groups=1)
            if self.single_snow:
                vidx = torch.randperm(snow_layer.shape[0])
                vidx = vidx[:int(snow_layer.shape[0]/2)]
                snow_layer = hsnow
                snow_layer[vidx] = vsnow[vidx]
            elif vertical_snow:
                snow_layer = vsnow
            else:
                snow_layer = hsnow
            self.snow.append(snow_layer)
            self.snow_rot.append(torch.rot90(snow_layer, k=2, dims=[2,3]))
        
        if not self.random_snow:
            np.random.set_state(rstate)

    @torch.no_grad()
    def total_forward(self, x_in):
        return self.forward(None, grad, self.num_timesteps-1, og=x_in)
    
    @torch.no_grad()
    def forward(self, x,grad=None, i=0, og=None):
        og_r = (og + 1.) / 2.
        og_gray = rgb_to_grayscale(og_r) * 1.5 + 0.5
        og_gray = torch.maximum(og_r, og_gray)
        br_coef = self.br_coef_list[i]
        scaled_og = br_coef * og_r + (1 - br_coef) * og_gray
        if self.fix_brightness:
            snowy_img = torch.clip(og_r + self.snow[i].cuda() + self.snow_rot[i].cuda(), 0.0, 1.0)
        else:
            snowy_img = torch.clip(scaled_og + self.snow[i].cuda() + self.snow_rot[i].cuda(), 0.0, 1.0)
        return (snowy_img * 2.) - 1.

class FGSMAttack(ForwardProcessBase):
    def __init__(self, device, min_epsilon, max_epsilon, num_timesteps, batch_size=32):
        self.device = device
        self.epsilons = np.linspace(min_epsilon, max_epsilon, num_timesteps).tolist()
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size
        # self.classifier = ResNetClassifier(model_path=r'snowification\helpers\resnet18.pt')

    @torch.no_grad()
    def reset_parameters(self, batch_size=-1):
        if batch_size != -1:
            self.batch_size = batch_size

    @torch.no_grad()
    def total_forward(self, x_in, grad=None):
        return self.forward(x=None, grad=grad, i=self.num_timesteps-1, og=x_in)

    @torch.no_grad()
    def forward(self, x, grad=None, i=0, og=None) -> Image:
        #for each of the images in the batch, we want to perturb the image by epsilon * sign(grad) then return all the perturbed images
        if grad is None:
            return og
        perturbed_images = []
        for j in range(og.shape[0]):
            # print("Image shape: ", {og[j].shape}, '\n')
            
            # print("Before: ",self.classifier.predict_image_class(image=image), '\n')
            
            preprocess = transforms.Compose([
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
            ])
            image_tensor = preprocess(og[j])
            
            # print("Grad shape: ", grad[j].shape, '\n')
            
            adv_tensor = image_tensor + self.epsilons[i] * torch.sign(grad[j])
            reverse_transform = transforms.Compose([
                transforms.Normalize(mean=[-0.4914/0.2471, -0.4822/0.2435, -0.4465/0.2616],
                                    std=[1/0.2471, 1/0.2435, 1/0.2616]),
                lambda x: x.clamp(0, 1)
            ])
            result = reverse_transform(adv_tensor)
            # print("After: ",self.classifier.predict_image_class(image_tensor=result.unsqueeze(0)), '\n')
            
            perturbed_images.append(result)
        return torch.stack(perturbed_images)
