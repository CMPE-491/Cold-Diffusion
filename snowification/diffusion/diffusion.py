import torch
from torch import nn
import torch.nn.functional as F

import os

from .forward_process_impl import DeColorization, Snow
from .color_utils import lab2rgb


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        device_of_kernel,
        channels = 3,
        timesteps = 1000,
        loss_type = 'l1',
        forward_process_type = 'Decolorization',
        train_routine = 'Final',
        sampling_routine='default',
        decolor_routine='Constant',
        decolor_ema_factor=0.9,
        decolor_total_remove=True,
        snow_level=1,
        random_snow=False,
        to_lab=False,
        recon_noise_std=0.0,
        load_snow_base=False,
        load_path=None,
        batch_size=32,
        single_snow=False,
        fix_brightness=False,
        results_folder=None,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.device_of_kernel = device_of_kernel
        

        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        self.train_routine = train_routine
        self.sampling_routine = sampling_routine

        self.snow_level = snow_level
        self.random_snow = random_snow
        self.batch_size = batch_size
        self.single_snow = single_snow

        self.to_lab = to_lab
        self.recon_noise_std = recon_noise_std
        # mixing coef of loss of pred vs ground_truth 

        # Gaussian Blur parameters
                                    
        if forward_process_type == 'Decolorization':
            self.forward_process = DeColorization(decolor_routine=decolor_routine,
                                                  decolor_ema_factor=decolor_ema_factor,
                                                  decolor_total_remove=decolor_total_remove,
                                                  channels=self.channels,
                                                  num_timesteps=self.num_timesteps,
                                                  to_lab=self.to_lab,)
        elif forward_process_type == 'Snow':
            if load_path is not None:
                snow_base_path = load_path.replace('model.pt', 'snow_base.npy')
                print(snow_base_path)
                load_snow_base = True
            else:
                snow_base_path = os.path.join(results_folder, 'snow_base.npy')
                load_snow_base = False
            self.forward_process = Snow(image_size=self.image_size,
                                        snow_level=self.snow_level, 
                                        random_snow=self.random_snow,
                                        num_timesteps=self.num_timesteps,
                                        snow_base_path=snow_base_path,
                                        batch_size=self.batch_size,
                                        single_snow=self.single_snow,
                                        load_snow_base=load_snow_base,
                                        fix_brightness=fix_brightness)
    
    @torch.no_grad()
    def sample_one_step(self, img, t, init_pred=None):

        x = self.prediction_step_t(img, t, init_pred)
        direct_recons = x.clone()
        
        if self.recon_noise_std > 0.0:
            self.recon_noise_std_array = torch.linspace(0.0, self.recon_noise_std, steps=self.num_timesteps)

        if self.train_routine in ['Final', 'Final_random_mean', 'Final_small_noise', 'Final_random_mean_and_actual']:

            if self.sampling_routine == 'default':

                x_times_sub_1 = x.clone()
                cur_time = torch.zeros_like(t)
                fp_index = torch.where(cur_time < t - 1)[0]
                for i in range(t.max() - 1):
                    x_times_sub_1[fp_index] = self.forward_process.forward(x_times_sub_1[fp_index], i, og=x[fp_index])
                    cur_time += 1
                    fp_index = torch.where(cur_time < t - 1)[0]

                x = x_times_sub_1


            elif self.sampling_routine == 'x0_step_down':
                
                x_times = x.clone()
                if self.recon_noise_std > 0.0:
                    x_times = x + torch.normal(0.0, self.recon_noise_std, size=x.size()).cuda()
                x_times_sub_1 = x_times.clone()

                cur_time = torch.zeros_like(t)
                fp_index = torch.where(cur_time < t)[0]
                for i in range(t.max()):
                    x_times_sub_1 = x_times.clone()
                    x_times[fp_index] = self.forward_process.forward(x_times[fp_index], i, og=x[fp_index])
                    cur_time += 1
                    fp_index = torch.where(cur_time < t)[0]


                x = img - x_times + x_times_sub_1

        elif self.train_routine == 'Step':
            img = x

        elif self.train_routine == 'Step_Gradient':
            x = img + x
        
        return x, direct_recons

    @torch.no_grad()
    def sample_multi_step(self, img, t_start, t_end):
        fp_index = torch.where(t_start > t_end)[0]
        img_new= img.clone()
        while len(fp_index) > 0:
            _, img_new_partial = self.sample_one_step(img_new[fp_index], t_start[fp_index])
            img_new[fp_index] = img_new_partial
            t_start = t_start - 1
            fp_index = torch.where(t_start > t_end)[0]
        return img_new


    @torch.no_grad()
    def sample(self, batch_size = 16, img=None, t=None):

        self.forward_process.reset_parameters(batch_size=batch_size)
        if t==None:
            t=self.num_timesteps

        og_img = img.clone()
        
        for i in range(t):
            with torch.no_grad():
                img = self.forward_process.forward(img, i, og=og_img)
        
        init_pred = None
        # 3(2), 2(1), 1(0)
        xt = img
        direct_recons = None
        while(t):
            step = torch.full((batch_size,), t - 1, dtype=torch.long).cuda()
            x, cur_direct_recons = self.sample_one_step(img, step, init_pred=init_pred)
            if direct_recons is None:
                direct_recons = cur_direct_recons
            img = x
            t = t - 1
        
        if self.to_lab:
            xt = lab2rgb(xt)
            direct_recons = lab2rgb(direct_recons)
            img = lab2rgb(img)
            
        return_dict =  {'xt': xt, 
                        'direct_recons': direct_recons, 
                        'recon': img,}
        return return_dict


    @torch.no_grad()
    def all_sample(self, batch_size=16, img=None, t=None, times=None, res_dict=None):
        
        self.forward_process.reset_parameters(batch_size=batch_size)
        if t == None:
            t = self.num_timesteps
        if times == None:
            times = t
        
        
        img_forward_list = []
        
        img_forward = img

        with torch.no_grad():
            img = self.forward_process.total_forward(img)

        X_0s = []
        X_ts = []

        init_pred = None
        while (times):
            step = torch.full((img.shape[0],), times - 1, dtype=torch.long).cuda()
            img, direct_recons = self.sample_one_step(img, step, init_pred=init_pred)
            X_0s.append(direct_recons.cpu())
            X_ts.append(img.cpu())
            times = times - 1

        X_0s.append(img_forward.cpu())
        X_ts.append(img_forward.cpu())

        init_pred_clone = None
        if init_pred is not None:
            init_pred_clone = init_pred.clone().cpu()
        if self.to_lab:
            for i in range(len(X_0s)):
                X_0s[i] = lab2rgb(X_0s[i])
                X_ts[i] = lab2rgb(X_ts[i])
            if init_pred is not None:
                init_pred_clone = lab2rgb(init_pred_clone)

        return X_0s, X_ts, init_pred_clone, img_forward_list

    def q_sample(self, x_start, t, return_total_blur=False):
        # So at present we will for each batch blur it till the max in t.
        # And save it. And then use t to pull what I need. It is nothing but series of convolutions anyway.
        # Remember to do convs without torch.grad
        
        final_sample = x_start.clone()
        
        max_iters = torch.max(t)
        all_blurs = []
        x = x_start[torch.where(t != -1)]
        blurring_batch_size = x.shape[0]
        if blurring_batch_size == 0:
            return final_sample

        for i in range(max_iters+1):
            with torch.no_grad():
                x = self.forward_process.forward(x, i, og=final_sample[torch.where(t != -1)])
                all_blurs.append(x)

                if i == max_iters:
                    total_blur = x.clone()

        all_blurs = torch.stack(all_blurs)

        choose_blur = []
        # step is batch size as well so for the 49th step take the step(batch_size)
        for step in range(blurring_batch_size):
            if step != -1:
                choose_blur.append(all_blurs[t[step], step])
            else:
                choose_blur.append(x_start[step])

        choose_blur = torch.stack(choose_blur)
        #choose_blur = all_blurs

        final_sample[torch.where(t != -1)] = choose_blur

        if return_total_blur:
            final_sample_total_blur = final_sample.clone()
            final_sample_total_blur[torch.where(t != -1)] = total_blur
            return final_sample, final_sample_total_blur
        return final_sample
    
    def loss_func(self, pred, true):
        if self.loss_type == 'l1':
            return (pred - true).abs().mean()
        elif self.loss_type == 'l2':
            return F.mse_loss(pred, true)
        elif self.loss_type == 'sqrt':
            return (pred - true).abs().mean().sqrt()
        else:
            raise NotImplementedError()

    
    
    def prediction_step_t(self, img, t, init_pred=None):
        return self.denoise_fn(img, t)

    def p_losses(self, x_start, t, t_pred=None):
        b, c, h, w = x_start.shape
        
        self.forward_process.reset_parameters()

        if self.train_routine == 'Final':
            x_blur, x_total_blur = self.q_sample(x_start=x_start, t=t, return_total_blur=True)

            x_recon = self.denoise_fn(x_blur, t)
            loss = self.loss_func(x_start, x_recon)
            
        elif self.train_routine == 'Step_Gradient':
            x_blur, x_total_blur = self.q_sample(x_start=x_start, t=t, return_total_blur=True)
            x_blur_sub = self.q_sample(x_start=x_start, t=t-1)

            x_blur_diff = x_blur_sub - x_blur
            x_blur_diff_pred = self.denoise_fn(x_blur, t)
            loss = self.loss_func(x_blur_diff, x_blur_diff_pred)

        elif self.train_routine == 'Step':
            x_blur, x_total_blur = self.q_sample(x_start=x_start, t=t, return_total_blur=True)
            x_blur_sub = self.q_sample(x_start=x_start, t=t-1)
            
            x_blur_sub_pred = self.denoise_fn(x_blur, t)
            loss = self.loss_func(x_blur_sub, x_blur_sub_pred)

        return loss

    def forward(self, x, *args, **kwargs):
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        if type(img_size) is tuple:
            img_w, img_h = img_size
        else:
            img_h, img_w = img_size, img_size
        assert h == img_h and w == img_w, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        t_pred = [] 
        for i in range(b):
            t_pred.append(torch.randint(0, t[i]+1, ()).item())
        t_pred = torch.Tensor(t_pred).to(device).long()  -1
        t_pred[t_pred < 0] = 0

        return self.p_losses(x, t, t_pred, *args, **kwargs)

    @torch.no_grad()
    def forward_and_backward(self, batch_size=16, img=None, t=None, times=None, eval=True):

        self.denoise_fn.eval()

        if t == None:
            t = self.num_timesteps

        Forward = []
        Forward.append(img)

        for i in range(t):
            with torch.no_grad():
                step = torch.full((batch_size,), i, dtype=torch.long, device=img.device)
                n_img = self.q_sample(x_start=img, t=step)
                Forward.append(n_img)

        Backward = []
        img = n_img
        while (t):
            step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)
            x1_bar = self.denoise_fn(img, step)

            Backward.append(img)

            xt_bar = x1_bar
            if t != 0:
                xt_bar = self.q_sample(x_start=xt_bar, t=step)

            xt_sub1_bar = x1_bar
            if t - 1 != 0:
                step2 = torch.full((batch_size,), t - 2, dtype=torch.long, device=img.device)
                xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, t=step2)

            x = img - xt_bar + xt_sub1_bar
            img = x
            t = t - 1

        return Forward, Backward, img