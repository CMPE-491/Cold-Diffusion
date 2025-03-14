from diffusion import get_dataset
from diffusion.diffusion import GaussianDiffusion
from diffusion.tester import Tester
import os
import argparse
from diffusion.model.get_model import get_model
from Fid.fid_score import calculate_fid_given_samples
import torch
import random

parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', default=50, type=int)
parser.add_argument('--sample_steps', default=None, type=int)
parser.add_argument('--save_folder_train', default='/cmlscratch/hmchu/cold_diff/', type=str)
parser.add_argument('--save_folder_test', default='/cmlscratch/hmchu/cold_diff_paper_test/', type=str)
parser.add_argument('--load_path', default=None, type=str)
parser.add_argument('--test_type', default='train_data', type=str)
parser.add_argument('--train_routine', default='Final', type=str)
parser.add_argument('--sampling_routine', default='x0_step_down', type=str)
parser.add_argument('--remove_time_embed', action="store_true")

parser.add_argument('--exp_name', default='', type=str)
parser.add_argument('--test_postfix', default='', type=str)
parser.add_argument('--dataset_folder', default='./root_cifar10', type=str)
parser.add_argument('--output_mean_scale', action='store_true')
parser.add_argument('--random_aug', action='store_true')
parser.add_argument('--model', default='UnetConvNext', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)

parser.add_argument('--forward_process_type', default='GaussianBlur')
# GaussianBlur args

# Decolor args
parser.add_argument('--decolor_routine', default='Constant')
parser.add_argument('--decolor_ema_factor', default=0.9, type=float)
parser.add_argument('--decolor_total_remove', action='store_true')

parser.add_argument('--load_model_steps', default=-1, type=int)
parser.add_argument('--resume_training', action='store_true')

parser.add_argument('--order_seed', default=-1.0, type=float)
parser.add_argument('--resolution', default=-1, type=int)

parser.add_argument('--to_lab', action='store_true')
parser.add_argument('--snow_level', default=1, type=int)
parser.add_argument('--random_snow', action='store_true')
parser.add_argument('--single_snow', action='store_true')
parser.add_argument('--fix_brightness', action='store_true')

parser.add_argument('--test_fid', action='store_true')

args = parser.parse_args()

if args.load_model_steps != -1:
    args.load_path = os.path.join(args.save_folder_train, f'model_{args.load_model_steps}.pt')
else:
    args.load_path = os.path.join(args.save_folder_train, 'model.pt')
print(f"Args: {args}")

args.save_folder_test = os.path.join(args.save_folder_test, args.test_type)
print(f"Save folder: {args.save_folder_test}")

dataset_folder = args.dataset_folder
print(f"Dataset folder: {dataset_folder}")

with_time_emb = not args.remove_time_embed

model = get_model(args, with_time_emb=with_time_emb).cuda()
model_one_shot = get_model(args, with_time_emb=False).cuda()

print(f"Dataset: {args.dataset}")
image_size = get_dataset.get_image_size(args.dataset)
if args.resolution != -1:
    image_size = (args.resolution, args.resolution)
print(f"Image size: {image_size}")

use_torchvison_dataset = False
if image_size[0] <= 64:
    train_batch_size = 32
elif image_size[0] > 64:
    train_batch_size = 16

seed_value=args.order_seed
torch.manual_seed(seed_value) # cpu vars
random.seed(seed_value) # Python
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


diffusion = GaussianDiffusion(
    model,
    image_size = image_size,
    device_of_kernel = 'cuda',
    channels = 3,
    timesteps = args.time_steps,   # number of steps
    loss_type = 'l1',    # L1 or L2
    train_routine=args.train_routine,
    sampling_routine = args.sampling_routine,
    forward_process_type=args.forward_process_type,
    decolor_routine=args.decolor_routine,
    decolor_ema_factor=args.decolor_ema_factor,
    decolor_total_remove=args.decolor_total_remove,
    snow_level=args.snow_level,
    random_snow=args.random_snow,
    single_snow=args.single_snow,
    batch_size=train_batch_size,
    to_lab=args.to_lab,
    load_snow_base=False,
    fix_brightness=args.fix_brightness,
    load_path = args.load_path,
    results_folder = args.save_folder_test,
).cuda()

tester = Tester(
    diffusion,
    dataset_folder,
    image_size = image_size,
    train_batch_size = train_batch_size,
    train_lr = 2e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,                       # turn on mixed precision training with apex
    results_folder = args.save_folder_test,
    load_path = args.load_path,
    random_aug=args.random_aug,
    torchvision_dataset=use_torchvison_dataset,
    dataset=f'{args.dataset}',
    order_seed=args.order_seed,
    to_lab=args.to_lab,
)

if __name__ == '__main__':
    if args.test_type == 'train_data':
        tester.test_from_data('train', s_times=args.sample_steps)
        if args.test_fid:
            tester.fid_distance_decrease_from_manifold(calculate_fid_given_samples, start=0, end=1000)

    elif args.test_type == 'test_data':
        tester.test_from_data('test', s_times=args.sample_steps)
        if args.test_fid:
            tester.fid_distance_decrease_from_manifold(calculate_fid_given_samples, start=0, end=2000)

    elif args.test_type == 'test_paper':
        tester.paper_invert_section_images() 
        if args.test_fid:
            tester.fid_distance_decrease_from_manifold(calculate_fid_given_samples, start=0, end=1000)

    elif args.test_type == 'test_paper_series':
        tester.paper_showing_diffusion_images()

    elif args.test_type == 'test_rebuttal':
        tester.paper_showing_diffusion_images_cover_page()
