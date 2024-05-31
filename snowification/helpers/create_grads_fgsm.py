import torchvision
import torch
import os
import shutil
from resnet_classifier import ResNetClassifier
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='./cifar_10_gradients', type=str)
parser.add_argument('--model_path', default='./resnet18.pt', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--data_type', default='train', type=str)
args = parser.parse_args()

classifier = ResNetClassifier(model_path=args.model_path)

def save_image_grads(root, is_train=True, batch_size=32):
    dataset = torchvision.datasets.CIFAR10(root='./data', train=is_train, download=True, transform=transforms.ToTensor())
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root)

    class_counts = dict()
    
    device = classifier.device
    classifier.model.to(device)
    classifier.model.eval()
    
    for batch_idx, (img_tensors, label_tensors) in enumerate(tqdm(data_loader)):
        img_tensors, label_tensors = img_tensors.to(device), label_tensors.to(device)

        # Process each image in the batch
        for i in range(img_tensors.size(0)):
            original_img = transforms.ToPILImage()(img_tensors[i].cpu()).convert("RGB")
            label = classifier.classes[label_tensors[i].item()]
            class_counts[label] = class_counts.get(label, 0) + 1

            grad = classifier.get_image_grad(original_img, label_tensors[i].item())
            #grad_img = transforms.ToPILImage()(grad.cpu()).convert("RGB")
            grad_img_path = os.path.join(root, f"{label}_{class_counts[label]}_grad.pt")
            #grad.save(grad_img_path)
            torch.save(grad, grad_img_path)
    
    print(f"Gradients for CIFAR-10 {'train' if is_train else 'test'} dataset created.")

if __name__ == "__main__":
    if args.data_type == "train":
        save_image_grads(root=args.root, is_train=True, batch_size=args.batch_size)
    elif args.data_type == "test":
        save_image_grads(root=args.root, is_train=False, batch_size=args.batch_size)
