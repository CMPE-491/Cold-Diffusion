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
                
def cycle_with_label(dl):
    while True:
        for data in dl:
            yield data
