import os
import torch
import torch.nn.functional as F
import time
import cv2
import copy
import numpy as np

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils

import numpy as np
import imageio
from .get_dataset import Dataset, get_dataset, CustomCIFAR10Dataset
from .color_utils import rgb2lab
from .utils import create_folder, cycle, cycle_with_label, custom_collate_fn
from .EMA_model import EMA

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False


class Tester(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        grad_folder,
        *,
        ema_decay = 0.995,
        image_size = 128,
        train_batch_size = 32,
        train_lr = 2e-5,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        fp16 = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 5000,
        save_with_time_stamp_every = 50000,
        results_folder = './results',
        load_path = None,
        random_aug=False,
        torchvision_dataset=False,
        dataset = None,
        to_lab=False,
        order_seed=-1,
    ):
        super().__init__()
        self.model = diffusion_model
        self.num_timesteps = diffusion_model.num_timesteps
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every
        self.save_with_time_stamp_every = save_with_time_stamp_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.to_lab = to_lab
        self.order_seed = order_seed

        self.random_aug = random_aug
        if torchvision_dataset:
            self.ds = get_dataset(dataset, folder, self.image_size, random_aug=random_aug)
        else:
            if(self.model.forward_process_type == 'FGSM'):
                self.ds = CustomCIFAR10Dataset(folder, grad_folder, image_size, random_aug=self.random_aug)
                self.data_loader = data.DataLoader(self.ds, batch_size = train_batch_size, collate_fn=custom_collate_fn, shuffle=True, pin_memory=True, num_workers=4)
            else:
                self.ds = Dataset(folder, image_size, random_aug=self.random_aug)
                self.data_loader = data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True, num_workers=4)
        
        post_process_func = lambda x: x
        if self.to_lab:
            post_process_func = rgb2lab
        
        self.post_process_func = post_process_func
        if(self.model.forward_process_type == 'FGSM'):
            self.dl = cycle_with_label(self.data_loader)
        else:
            self.dl = cycle(self.data_loader, f=post_process_func)

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0


        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'

        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt, opt_level='O1')

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok = True)

        self.reset_parameters()

        if load_path != None:
            self.load(load_path)
    
    def _process_item(self, x):
        f = self.post_process_func
        if type(x) == list:
            return f(x[0])
        else:
            return f(x)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, save_with_time_stamp=False):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        if save_with_time_stamp:
            torch.save(data, str(self.results_folder / f'model_{self.step}.pt'))
        else:
            torch.save(data, str(self.results_folder / f'model.pt'))

    def load(self, load_path):
        print("Loading: ", load_path)
        data = torch.load(load_path)

        self.step = data['step']
        print("Model at step: ", self.step)
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def add_title(self, path, title_texts):
        img1 = cv2.imread(path)

        # --- Here I am creating the border---
        black = [0, 0, 0]  # ---Color of the border---
        constant = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
        #cv2.imshow('constant', constant)
        height = 20
        violet = np.zeros((height, constant.shape[1], 3), np.uint8)
        violet[:] = (255, 0, 180)

        vcat = cv2.vconcat((violet, constant))
        #cv2.imshow('vcat', vcat)

        font = cv2.FONT_HERSHEY_SIMPLEX
        
        title_counts = len(title_texts)
        for i, title in enumerate(title_texts):
            vertical_pos = i * (violet.shape[1] // title_counts) + (violet.shape[1] // (title_counts * 2))
            cv2.putText(vcat, str(title), (vertical_pos, height-2), font, 0.5, (0, 0, 0), 1, 0)
        cv2.imwrite(path, vcat)
 
    
    def save_test_images(self, X_ts, batch_size: int, batch_idx: int):
        to_PIL = transforms.ToPILImage()
        
        dir_path = self.results_folder
        original_path = dir_path / "original"
        snowified_path = dir_path / "snowified"
        cleaned_path = dir_path / "cleaned"
        os.makedirs(dir_path, exist_ok=True)
        os.makedirs(original_path, exist_ok=True)
        os.makedirs(snowified_path, exist_ok=True)
        os.makedirs(cleaned_path, exist_ok=True)
        
        for i in range(len(X_ts)):
            if (i != 0) and (i != len(X_ts) - 2) and (i != len(X_ts) - 1):
                continue
            
            x_t = X_ts[i]
            for j, image in enumerate(x_t):
                # Normalize and convert to PIL image
                image = (image + 1) * 0.5
                pil_img = to_PIL(image.cpu())

                if i == len(X_ts) - 1:
                    image_path = dir_path / "original" / f"{batch_size*batch_idx+j}_original.png"
                elif i != len(X_ts) - 2:
                    image_path = dir_path / "snowified" / f"{batch_size*batch_idx+j}_snow.png"
                else:
                    image_path = dir_path / "cleaned" / f"{batch_size*batch_idx+j}_cleaned.png"
                
                pil_img.save(str(image_path))

    def save_gif(self, X_0s, X_ts, extra_path, init_recon=None, og=None):

        frames_t = []
        frames_0 = []
        to_PIL = transforms.ToPILImage()
        
        if init_recon is not None:
            init_recon = (init_recon + 1) * 0.5
        self.gif_len = len(X_0s)
        for i in range(len(X_0s)):

            print(i)
            start_time = time.time()
            x_0 = X_0s[i]
            x_0 = (x_0 + 1) * 0.5
            x_0_grid = utils.make_grid(x_0, nrow=6)
            if init_recon is not None:
                init_recon_grid = utils.make_grid(init_recon, nrow=6)
                x_0_grid = utils.make_grid(torch.stack((x_0_grid, og, init_recon_grid)), nrow=3)
                title_texts = [str(i), 'og', 'init_recon']
            elif og is not None:
                x_0_grid = utils.make_grid(torch.stack((x_0_grid, og)), nrow=2)
                title_texts = [str(i), 'og']
            utils.save_image(x_0_grid, str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'))
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'), title_texts)
            frames_0.append(imageio.imread(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png')))


            x_t = X_ts[i]
            all_images = (x_t + 1) * 0.5
            all_images_grid = utils.make_grid(all_images, nrow=6)

            if init_recon is not None:
                init_recon_grid = utils.make_grid(init_recon, nrow=6)
                all_images_grid = utils.make_grid(torch.stack((all_images_grid, og, init_recon_grid)), nrow=3)
            elif og is not None:
                all_images_grid = utils.make_grid(torch.stack((all_images_grid, og)), nrow=2)
            utils.save_image(all_images_grid, str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'))
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'), title_texts)
            frames_t.append(imageio.imread(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png')))


        imageio.mimsave(str(self.results_folder / f'Gif-{extra_path}-x0.gif'), frames_0)
        imageio.mimsave(str(self.results_folder / f'Gif-{extra_path}-xt.gif'), frames_t)

    
    def save_og_test(self, og_dict, extra_path):
        for k, img in og_dict.items():
            img_scale = (img + 1) * 0.5
            img_grid = utils.make_grid(img_scale, nrow=6)
            utils.save_image(img_grid, str(self.results_folder / f'{k}-{extra_path}.png'))
            self.add_title(str(self.results_folder / f'{k}-{extra_path}.png'), '{k}')
            og_dict[k] = img_grid
    
    def test_from_data(self, extra_path, s_times=None): 

        for batch_idx, x in enumerate(self.data_loader):
            if(self.model.forward_process_type == 'FGSM'):
                og_img, grad = x
                og_img = self._process_item(og_img).cuda()
                grad = grad.cuda()
            else:
                og_img = self._process_item(x).cuda()
            og_dict = {'og': og_img.cuda()}
            X_0s, X_ts, init_recon, img_forward_list = self.ema_model.all_sample(batch_size=self.batch_size, img=og_img, grad =grad, times=s_times, res_dict=og_dict)
            print(f'Testing on batch {batch_idx+1}/{len(self.data_loader)}')
            og_dict = {'og': og_img.cpu()}
            
            self.save_test_images(X_ts, self.batch_size, batch_idx)
            # self.save_og_test(og_dict, extra_path)
            #self.save_gif(X_0s, X_ts, extra_path, init_recon=init_recon, og=og_dict['og'])

            if og_img.shape[0] != self.batch_size:
                continue

        print(f"'test_from_data' finished. Results saved in '{self.results_folder}'")

    def paper_invert_section_images(self, s_times=None):

        cnt = 0
        max_pixel_diff = 0.0 
        max_pixel_diff_sum = -1.0
        for i in range(20):
            batches = self.batch_size
            og_img = next(self.dl).cuda()
            print(og_img.shape)

            X_0s, X_ts, _, _ = self.ema_model.all_sample(batch_size=batches, img=og_img, times=s_times)
            og_img = (og_img + 1) * 0.5

            for j in range(og_img.shape[0]//9):
                original = og_img[j: j + 9]
                utils.save_image(original, str(self.results_folder / f'original_{cnt}.png'), nrow=3)

                direct_recons = X_0s[0][j: j + 9]
                direct_recons = (direct_recons + 1) * 0.5
                utils.save_image(direct_recons, str(self.results_folder / f'direct_recons_{cnt}.png'), nrow=3)

                sampling_recons = X_0s[-1][j: j + 9]
                sampling_recons = (sampling_recons + 1) * 0.5
                utils.save_image(sampling_recons, str(self.results_folder / f'sampling_recons_{cnt}.png'), nrow=3)

                diff = (direct_recons - sampling_recons).squeeze().sum(dim=0).max()
                diff_sum = (direct_recons - sampling_recons).squeeze().sum(dim=0).sum()
                if diff > max_pixel_diff:
                    max_pixel_diff = diff

                if diff_sum > max_pixel_diff_sum:
                    max_pixel_diff_sum = diff_sum

                blurry_image = X_ts[0][j: j + 9]
                blurry_image = (blurry_image + 1) * 0.5
                utils.save_image(blurry_image, str(self.results_folder / f'blurry_image_{cnt}.png'), nrow=3)




                blurry_image = cv2.imread(f'{self.results_folder}/blurry_image_{cnt}.png')
                direct_recons = cv2.imread(f'{self.results_folder}/direct_recons_{cnt}.png')
                sampling_recons = cv2.imread(f'{self.results_folder}/sampling_recons_{cnt}.png')
                original = cv2.imread(f'{self.results_folder}/original_{cnt}.png')

                #black = [255, 255, 255]
                black = [0, 0, 0]
                blurry_image = cv2.copyMakeBorder(blurry_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
                direct_recons = cv2.copyMakeBorder(direct_recons, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
                sampling_recons = cv2.copyMakeBorder(sampling_recons, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
                original = cv2.copyMakeBorder(original, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)

                im_h = cv2.hconcat([blurry_image, direct_recons, sampling_recons, original])
                cv2.imwrite(f'{self.results_folder}/all_{cnt}.png', im_h)

                cnt += 1 


    def paper_showing_diffusion_images(self, s_times=None):

        cnt = 0
        to_show = [0, 1, 2, 4, 8, 16, 24, 32, 40, 44, 46, 48, 49]

        for i in range(5):
            batches = self.batch_size
            og_img = next(self.dl).cuda()
            print(og_img.shape)

            X_0s, X_ts, _, _ = self.ema_model.all_sample(batch_size=batches, img=og_img, times=s_times)
            og_img = (og_img + 1) * 0.5

            for k in range(X_ts[0].shape[0]):
                l = []

                for j in range(len(X_ts)):
                    x_t = X_ts[j][k]
                    x_t = (x_t + 1) * 0.5
                    utils.save_image(x_t, str(self.results_folder / f'x_{len(X_ts)-j}_{cnt}.png'), nrow=1)
                    x_t = cv2.imread(f'{self.results_folder}/x_{len(X_ts)-j}_{cnt}.png')
                    if j in to_show:
                        l.append(x_t)


                x_0 = X_0s[-1][k]
                x_0 = (x_0 + 1) * 0.5
                utils.save_image(x_0, str(self.results_folder / f'x_best_{cnt}.png'), nrow=1)
                x_0 = cv2.imread(f'{self.results_folder}/x_best_{cnt}.png')
                l.append(x_0)
                im_h = cv2.hconcat(l)
                cv2.imwrite(f'{self.results_folder}/all_{cnt}.png', im_h)

                cnt+=1

    def paper_showing_diffusion_images_cover_page(self):

        cnt = 0
        to_show = [int(self.num_timesteps * i / 4) for i in range(4)]
        to_show.append(self.num_timesteps - 1)

        for i in range(50):
            batches = self.batch_size
            og_img = next(self.dl).cuda()
            print(og_img.shape)

            Forward, Backward, final_all = self.ema_model.forward_and_backward(batch_size=batches, img=og_img)
            og_img = (og_img + 1) * 0.5
            final_all = (final_all + 1) * 0.5

            for k in range(Forward[0].shape[0]):
                l = []

                utils.save_image(og_img[k], str(self.results_folder / f'og_img_{cnt}.png'), nrow=1)
                start = cv2.imread(f'{self.results_folder}/og_img_{cnt}.png')
                l.append(start)

                for j in range(len(Forward)):
                    x_t = Forward[j][k]
                    x_t = (x_t + 1) * 0.5
                    utils.save_image(x_t, str(self.results_folder / f'temp.png'), nrow=1)
                    x_t = cv2.imread(f'{self.results_folder}/temp.png')
                    if j in to_show:
                        l.append(x_t)

                for j in range(len(Backward)):
                    x_t = Backward[j][k]
                    x_t = (x_t + 1) * 0.5
                    utils.save_image(x_t, str(self.results_folder / f'temp.png'), nrow=1)
                    x_t = cv2.imread(f'{self.results_folder}/temp.png')
                    if (len(Backward) - j) in to_show:
                        l.append(x_t)


                utils.save_image(final_all[k], str(self.results_folder / f'final_{cnt}.png'), nrow=1)
                final = cv2.imread(f'{self.results_folder}/final_{cnt}.png')
                l.append(final)


                im_h = cv2.hconcat(l)
                cv2.imwrite(f'{self.results_folder}/all_{cnt}.png', im_h)

                cnt+=1 
