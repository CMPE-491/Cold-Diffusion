from pathlib import Path

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

def loss_backwards(amp, fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)