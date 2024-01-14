import copy
import torch
from functools import partial
import time

from torch.utils import data
from pathlib import Path
from torch.optim import Adam

from .get_dataset import get_dataset, Dataset
from .color_utils import rgb2lab, lab2rgb
from .EMA_model import EMA
from .utils import cycle

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

# helper function for loss_backwards
def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
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

        self.random_aug = random_aug
        if torchvision_dataset:
            self.ds = get_dataset(dataset, folder, self.image_size, random_aug=random_aug)
        else:
            self.ds = Dataset(folder, image_size, random_aug=self.random_aug)
        post_process_func = lambda x: x
        if self.to_lab:
            post_process_func = rgb2lab
        
        self.data_loader = data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True, num_workers=4)

        
        self.post_process_func = post_process_func
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
        print("Loading : ", load_path)
        data = torch.load(load_path)

        self.step = data['step']
        print("Model at step : ", self.step)
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def train(self):
        backwards = partial(loss_backwards, self.fp16)
        start_time = time.time()
        
        while self.step < self.train_num_steps:

            for i in range(self.gradient_accumulate_every):
                data = next(self.dl).cuda()
                loss = self.model(data)
                print(f'{self.step}: {loss.item()}')
                backwards(loss / self.gradient_accumulate_every, self.opt)

            if self.step != 0 and self.step % 100 == 0:
                print(f'time for 100 steps: {time.time() - start_time}')
                start_time = time.time()

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                #if 'flower' not in args.dataset or 'cifar':
                """
                if True:
                    milestone = self.step // self.save_and_sample_every
                    batches = self.batch_size
                    og_img = next(self.dl).cuda()
                    sample_dict = self.ema_model.sample(batch_size=batches, img=og_img)
                    if self.to_lab:
                        og_img = lab2rgb(og_img)
                    sample_dict['og'] = og_img

                    print(f'images saved: {sample_dict.keys()}')
                    for k, img in sample_dict.items():
                        img_scale = (img + 1) * 0.5
                        utils.save_image(img_scale, str(self.results_folder / f'sample-{k}-{milestone}.png'), nrow=6)
                """
                self.save()
            # Save model with time stamp
            if self.step != 0 and self.step % self.save_with_time_stamp_every == 0:
                self.save(save_with_time_stamp=True)

            self.step += 1

        self.save()
        print('training completed')
