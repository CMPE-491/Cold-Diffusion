import torch
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch import nn, optim
import torch.nn.functional as F


import numpy as np

class AdversarialAttack:
    def __init__(self):
        self.image_size = 32
        self.model_path = '/content/drive/MyDrive/cifar10_resnet18.pth'
        self.train_path = '/content/drive/MyDrive/root_cifar10/'
        self.test_path = '/content/drive/MyDrive/root_cifar10_test/'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def preprocessing(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.train_dataset = ImageFolder(root=self.train_path, transform=transform)
        self.test_dataset = ImageFolder(root=self.test_path, transform=transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)

    def generate_model(self):
        self.preprocessing()
        self.model = models.resnet18(num_classes=10,pretrained=False).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        
        num_epochs = 3
        # Train the model
        for epoch in range(num_epochs):
            self.model.train()  # Set model to training mode
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            print(f'Epoch {epoch + 1}, Loss: {running_loss/len(self.train_loader)}')

        print('Finished Training')

        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)

        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        for param in self.model.fc.parameters():
          param.requires_grad = True

        self.model.to(self.device)
        print(f"model loaded: {self.model_path}")

    def generate_adversary(self, image, label):
      transform = transforms.Compose([transforms.ToTensor()])
      
      # Convert the input image to a PyTorch tensor
      image = transform(image).unsqueeze(0).to(self.device)

      # Convert the label to a PyTorch tensor and reshape it
      label = torch.from_numpy(label)
      label = label.view(1, -1).to(self.device)

      # Set requires_grad to True before the forward pass
      image.requires_grad = True

      self.model.eval()

      # Forward pass
      prediction = self.model(image).to(self.device)

      # Compute MSE loss
      loss = F.cross_entropy(prediction, label.argmax(dim=1))

      # Zero the gradient before computing the gradient
      self.model.zero_grad()

      # Ensure that image requires gradients before computing gradients
      if image.grad is not None:
          image.grad.zero_()

      # Compute gradient
      gradient = torch.autograd.grad(loss, image)[0]
      sign_grad = torch.sign(gradient)

      return sign_grad
    
    def generate_adversary_pgd(self, image, label, epsilon, num_iter):
      alpha = epsilon / num_iter
      transform = transforms.Compose([transforms.ToTensor()])
      
      # Convert the input image to a PyTorch tensor and clone to create a leaf variable
      original_image = transform(image).unsqueeze(0).to(self.device)
      perturbed_image = original_image.clone()
      perturbed_image.requires_grad = True

      # Convert the label to a PyTorch tensor and reshape it
      label = torch.from_numpy(label)
      label = label.view(1, -1).to(self.device)

      for i in range(num_iter):
          self.model.eval()

          # Forward pass
          prediction = self.model(perturbed_image).to(self.device)

          # Compute loss
          loss = F.cross_entropy(prediction, label.argmax(dim=1))

          # Zero the gradient
          self.model.zero_grad()

          # Compute gradient
          loss.backward()
          gradient = perturbed_image.grad.data

          # Apply perturbation
          perturbed_image = perturbed_image + alpha * gradient.sign()
          perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Clamp to ensure values are within [0, 1]

          # Re-clone the perturbed image and set requires_grad to True for next iteration
          perturbed_image = perturbed_image.detach().clone()
          perturbed_image.requires_grad = True

      # Calculate final perturbation
      final_perturbation = perturbed_image - original_image
      return final_perturbation
