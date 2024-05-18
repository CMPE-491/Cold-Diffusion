import torchvision
import os
import shutil
from resnet_classifier import ResNetClassifier
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='./cifar_10_dataset', type=str)
parser.add_argument('--model_path', default='./resnet18.pt', type=str)
parser.add_argument('--add_adversarial', action='store_true')
parser.add_argument('--attack_types', nargs='+', type=str, default=['FGSM', 'PGD'])
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--data_type', default='train', type=str)
args = parser.parse_args()


classifier = ResNetClassifier(model_path=args.model_path)

def create_cifar10_train(
    root,
    add_adversarial=False,
    attack_types: list[str] = ["FGSM", "PGD"],
    batch_size=32
):
    print("Creating CIFAR-10 train dataset")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    trainset_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=4)
    print("Trainset loaded")
    
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root)
    if add_adversarial:
        os.makedirs(os.path.join(root, "adv"), exist_ok=True)
        if "FGSM" in attack_types:
            os.makedirs(os.path.join(root, "adv", "fgsm"), exist_ok=True)
        if "PGD" in attack_types:
            os.makedirs(os.path.join(root, "adv", "pgd"), exist_ok=True)
    os.makedirs(os.path.join(root, "clean"), exist_ok=True)
    
    class_counts = dict() # {"airplane": 0, "automobile": 0, ...}

    device = classifier.device
    classifier.model.to(device)
    classifier.model.eval()

    for batch_idx, (img_tensors, label_tensors) in enumerate(trainset_loader):
        print(f"Creating batch {batch_idx+1}/{len(trainset_loader)}")
        
        img_tensors, label_tensors = img_tensors.to(device), label_tensors.to(device)

        # Process each image in the batch
        for i in range(img_tensors.size(0)):
            original_img = transforms.ToPILImage()(img_tensors[i].cpu()).convert("RGB")
            
            label = classifier.classes[label_tensors[i].item()]
            class_counts[label] = class_counts.get(label, 0) + 1

            original_img_path = os.path.join(root, "clean", f"{label}_{class_counts[label]}.png")
            original_img.save(original_img_path)

            if add_adversarial:
                if "FGSM" in attack_types:
                    adv_img = classifier.adversarial_attack(
                        image=original_img,
                        epsilon=10/255.0,
                        true_label=label_tensors[i].item()
                    )
                    adv_img_path = os.path.join(root, "adv", "fgsm", f"{label}_{class_counts[label]}_adv.png")
                    adv_img.save(adv_img_path)
                if "PGD" in attack_types:
                    adv_img = classifier.pgd_attack(
                        image=original_img,
                        epsilon=4/255.0,
                        num_steps=20,
                        true_label=label_tensors[i].item()
                    )
                    adv_img_path = os.path.join(root, "adv", "pgd", f"{label}_{class_counts[label]}_adv.png")
                    adv_img.save(adv_img_path)
    
    print("Trainset created.")

def create_cifar10_test(
    root,
    add_adversarial=False,
    attack_types: list[str] = ["FGSM", "PGD"], 
    batch_size=32
):
    print("Creating CIFAR-10 test dataset")
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    testset_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root)
    if add_adversarial:
        os.makedirs(os.path.join(root, "adv"), exist_ok=True)
        if "FGSM" in attack_types:
            os.makedirs(os.path.join(root, "adv", "fgsm"), exist_ok=True)
        if "PGD" in attack_types:
            os.makedirs(os.path.join(root, "adv", "pgd"), exist_ok=True)
    os.makedirs(os.path.join(root, "clean"), exist_ok=True)
    
    class_counts = dict() # {"airplane": 0, "automobile": 0, ...}

    device = classifier.device
    classifier.model.to(device)
    classifier.model.eval()

    for batch_idx, (img_tensors, label_tensors) in enumerate(testset_loader):
        print(f"Creating batch {batch_idx+1}/{len(testset_loader)}")
        
        img_tensors, label_tensors = img_tensors.to(device), label_tensors.to(device)

        # Process each image in the batch
        for i in range(img_tensors.size(0)):
            original_img = transforms.ToPILImage()(img_tensors[i].cpu()).convert("RGB")
            
            label = classifier.classes[label_tensors[i].item()]
            class_counts[label] = class_counts.get(label, 0) + 1

            original_img_path = os.path.join(root, "clean", f"{label}_{class_counts[label]}.png")
            original_img.save(original_img_path)

            if add_adversarial:
                if "FGSM" in attack_types:
                    adv_img = classifier.adversarial_attack(
                        image=original_img,
                        epsilon=10/255.0,
                        true_label=label_tensors[i].item()
                    )
                    adv_img_path = os.path.join(root, "adv", "fgsm", f"{label}_{class_counts[label]}_adv.png")
                    adv_img.save(adv_img_path)
                if "PGD" in attack_types:
                    adv_img = classifier.pgd_attack(
                        image=original_img,
                        epsilon=4/255.0,
                        num_steps=20,
                        true_label=label_tensors[i].item()
                    )
                    adv_img_path = os.path.join(root, "adv", "pgd", f"{label}_{class_counts[label]}_adv.png")
                    adv_img.save(adv_img_path)

    print("Testset created.")

if args.data_type == "train":
    create_cifar10_train(
        root=args.root,
        add_adversarial=args.add_adversarial,
        attack_types=args.attack_types,
        batch_size=args.batch_size
    )
elif args.data_type == "test":
    create_cifar10_test(
        root=args.root,
        add_adversarial=args.add_adversarial,
        attack_types=args.attack_types,
        batch_size=args.batch_size
    )