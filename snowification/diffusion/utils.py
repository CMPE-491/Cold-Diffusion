from pathlib import Path
import torch

def custom_collate_fn(batch):
    images = torch.stack([item.image for item in batch])
    grads = torch.stack([item.grad for item in batch])
    return images, grads

def create_folder(path):
    path_to_create = Path(path)
    path_to_create.mkdir(parents=True, exist_ok = True)

def cycle(dl, f=None):
    while True:
        for data in dl:
            # Temporary fix for torchvision CIFAR-10
            if type(data) == list:
                yield f(data[0])
            else:
                yield f(data)
                
def cycle_with_label(dl):
    while True:
        for data in dl:
            yield data
