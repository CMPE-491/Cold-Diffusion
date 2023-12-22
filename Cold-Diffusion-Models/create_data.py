import torchvision
import os
import sys
import errno
import shutil
import tensorflow as tf
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

sys.path.append("/content/drive/MyDrive/python_files")

from adversial_attack import AdversarialAttack

adv_attack = AdversarialAttack()
adv_attack.load_model()

epsilon = 0.05

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

def create_cifar10_train(root):
  trainset = torchvision.datasets.CIFAR10(
              root='./data', train=True, download=True)
  root = '/content/drive/MyDrive/root_cifar10/'
  del_folder(root)
  create_folder(root)

  for i in range(10):
      lable_root = root + str(i) + '/'
      create_folder(lable_root)

  for idx in range(len(trainset)):
      print(idx)
      img, label = trainset[idx]
      one_hot_label = tf.keras.utils.to_categorical(label, 10)

      perturbations = adv_attack.generate_adversary(img, one_hot_label).numpy()
      perturbations = perturbations.reshape(np.array(img).shape)
      adversarial = img + (perturbations * epsilon)

      adversarial = np.squeeze(adversarial)
      adversarial = Image.fromarray(adversarial.astype('uint8'))

      img.save(root + str(label) + '/' + str(idx) + '.png')      
      adversarial.save(root + str(label) + '/adv' + str(idx) + '.png')


def create_cifar10_test(root):
  trainset = torchvision.datasets.CIFAR10(
              root='./data', train=False, download=True)
  root = '/content/drive/MyDrive/root_cifar10_test/'
  del_folder(root)
  create_folder(root)
  
  for i in range(10):
      lable_root = root + str(i) + '/'
      create_folder(lable_root)
  
  for idx in range(len(trainset)):
      print(idx)
      img, label = trainset[idx]
      one_hot_label = tf.keras.utils.to_categorical(label, 10)

      perturbations = adv_attack.generate_adversary(img, one_hot_label).numpy()
      perturbations = perturbations.reshape(np.array(img).shape)
      adversarial = img + (perturbations * epsilon)

      adversarial = np.squeeze(adversarial)
      adversarial = Image.fromarray(adversarial.astype('uint8'))

      img.save(root + str(label) + '/' + str(idx) + '.png')      
      adversarial.save(root + str(label) + '/adv' + str(idx) + '.png')
      
create_cifar10_train("/content/drive/MyDrive/root_cifar10/")
create_cifar10_test("/content/drive/MyDrive/root_cifar10_test/")
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