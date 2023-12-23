import torchvision
import os
import sys
import errno
import shutil
import tensorflow as tf
from PIL import Image
import numpy as np

sys.path.append("/content/drive/MyDrive/python_files")

from adversarial_attack import AdversarialAttack

adv_attack = AdversarialAttack()
adv_attack.load_model()

epsilon = 3/255
num_iter = 5

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def del_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as exc:
        pass

"""
CelebA_folder = '/fs/cml-datasets/CelebA-HQ/images-128/' # change this to folder which has CelebA data

############################################# MNIST ###############################################
trainset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True)
root = '/content/drive/MyDrive/root_mnist/'
del_folder(root)
create_folder(root)

for i in range(10):
    lable_root = root + str(i) + '/'
    create_folder(lable_root)

for idx in range(len(trainset)):
    img, label = trainset[idx]
    print(idx)
    img.save(root + str(label) + '/' + str(idx) + '.png')

trainset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True)
root = '/content/drive/MyDrive/root_mnist_test/'
del_folder(root)
create_folder(root)

for i in range(10):
    lable_root = root + str(i) + '/'
    create_folder(lable_root)

for idx in range(len(trainset)):
    img, label = trainset[idx]
    print(idx)
    img.save(root + str(label) + '/' + str(idx) + '.png')
"""


############################################# Cifar10 ###############################################

def create_cifar10_train(root, add_adversarial=False):
  trainset = torchvision.datasets.CIFAR10(
              root='./data', train=True, download=True)
  del_folder(root)
  create_folder(root)

  for i in range(10):
      lable_root = root + str(i) + '/'
      create_folder(lable_root)

  for idx in range(len(trainset)):
    print(idx)
    img, label = trainset[idx]

    if add_adversarial:

        one_hot_label = tf.keras.utils.to_categorical(label, 10)

        perturbations = adv_attack.generate_adversary_pgd(img, one_hot_label, epsilon, num_iter)
        perturbations = perturbations.detach().numpy()  # Convert to NumPy array if it's a PyTorch tensor

        # Ensure perturbations are the same shape as img
        perturbations = perturbations.reshape(np.array(img).shape)

        adversarial = np.array(img) + perturbations
        adversarial = np.clip(adversarial, 0, 255).astype(np.uint8)
        
        adversarial = np.squeeze(adversarial)
        adversarial = Image.fromarray(adversarial.astype('uint8'))
        
        adversarial.save(root + str(label) + '/' + str(idx) + '.png')

    img.save(root + str(label) + '/' + str(idx) + '.png')     


def create_cifar10_test(root, add_adversarial=False):
  trainset = torchvision.datasets.CIFAR10(
              root='./data', train=False, download=True)
  del_folder(root)
  create_folder(root)
  
  for i in range(10):
      lable_root = root + str(i) + '/'
      create_folder(lable_root)
  
  for idx in range(len(trainset)):
    print(idx)
    img, label = trainset[idx]

    if add_adversarial:

        one_hot_label = tf.keras.utils.to_categorical(label, 10)

        perturbations = adv_attack.generate_adversary_pgd(img, one_hot_label, epsilon, num_iter)
        perturbations = perturbations.detach().numpy()  # Convert to NumPy array if it's a PyTorch tensor

        # Ensure perturbations are the same shape as img
        perturbations = perturbations.reshape(np.array(img).shape)

        adversarial = np.array(img) + perturbations
        adversarial = np.clip(adversarial, 0, 255).astype(np.uint8)
        
        adversarial = np.squeeze(adversarial)
        adversarial = Image.fromarray(adversarial.astype('uint8'))
        
        adversarial.save(root + str(label) + '/' + str(idx) + '.png')

    img.save(root + str(label) + '/' + str(idx) + '.png')    
      
create_cifar10_train("/content/drive/MyDrive/root_cifar10/",True)
create_cifar10_test("/content/drive/MyDrive/root_cifar10_test/",True)


"""
############################################# CelebA ###############################################
root_train = './root_celebA_128_train_new/'
root_test = './root_celebA_128_test_new/'
del_folder(root_train)
create_folder(root_train)

del_folder(root_test)
create_folder(root_test)

exts = ['jpg', 'jpeg', 'png']
folder = CelebA_folder
paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

for idx in range(len(paths)):
    img = Image.open(paths[idx])
    print(idx)
    if idx < 0.9*len(paths):
        img.save(root_train + str(idx) + '.png')
    else:
        img.save(root_test + str(idx) + '.png')
"""