from PIL import Image
from pathlib import Path
from torch.utils import data
from torchvision import transforms
from torchvision import datasets

def get_transform(image_size, random_aug=False, resize=False):
    if image_size[0] == 64:
        transform_list = [
            transforms.CenterCrop((128,128)),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ]
    elif not random_aug:
        transform_list = [
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ]
        if resize:
            transform_list = [transforms.Resize(image_size)] + transform_list
        T = transforms.Compose(transform_list)
    else:
        s = 1.0
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        T = transforms.Compose([
            transforms.RandomResizedCrop(size=image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    return T

def get_image_size(name):
    if 'cifar10' in name:
        return (32, 32)
    if 'celebA' in name:
        return (128, 128)
    if 'flower' in name:
        return (128, 128)

def get_dataset(name, folder, image_size, random_aug=False):
    print(folder)
    if name == 'cifar10_train':
        return datasets.CIFAR10(folder, train=True, transform=get_transform(image_size, random_aug=random_aug), download = True)
    if name == 'cifar10_test':
        return datasets.CIFAR10(folder, train=False, transform=get_transform(image_size, random_aug=random_aug), download = True)
    if name == 'CelebA_train':
        return datasets.CelebA(folder, split='train', transform=get_transform(image_size, random_aug=random_aug), download=True)
    if name == 'CelebA_test':
        return datasets.CelebA(folder, split='test', transform=get_transform(image_size, random_aug=random_aug), download = True)
    if name == 'flower_train':
        return datasets.Flowers102(folder, split='train', transform=get_transform(image_size, random_aug=random_aug, resize=True), download=True)
    if name == 'flower_test':
        return datasets.Flowers102(folder, split='test', transform=get_transform(image_size, random_aug=random_aug, resize=True), download=True)

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts = ['jpg', 'jpeg', 'png'], random_aug=False):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = get_transform(self.image_size, random_aug=random_aug)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

class Dataset_Cifar10(data.Dataset):
    def __init__(self, folder, image_size, exts = ['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)
    
CLASS_NAME_TO_INDEX = {
    'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3,
    'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
}

class CustomCIFAR10Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png'], random_aug=False):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(folder).glob(f'**/*.{ext}')]
        self.transform = get_transform(self.image_size, random_aug=random_aug)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        # Extract class name from the filename
        class_name = path.stem.split('_')[0]
        label = CLASS_NAME_TO_INDEX.get(class_name, -1)  # Use -1 for unknown class
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label