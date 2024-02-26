import os
from typing import Tuple
import torch
import torch.nn.functional as F
import torchvision.models as models

from PIL import Image
from torch import nn
from torchvision import transforms

from resnet import resnet18


class ResNetClassifier:        
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, model_path: str):
        self.model = self._load_resnet_model(model_path=model_path)

    def _load_resnet_model(self, model_path: str) -> models.ResNet:
        resnet = resnet18(pretrained=True, model_path=model_path, device=self.device)
        model = resnet.to(self.device)
        return model
    
    def preprocess_image(self, image: Image) -> torch.Tensor:
        """
        Preprocesses an image to be used as input for the ResNet model.
        
        Args:
            image(Image): PIL image
        Returns:
            torch.Tensor: Image tensor with shape (1, 3, 32, 32)
        """
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
        ])
        image_tensor = preprocess(image).unsqueeze(0)
        
        return image_tensor
    
    def reverse_preprocess_image(self, image_tensor: torch.Tensor) -> Image:
        """
        Reverses the preprocessing of an image.
        
        Args:
            image_tensor(torch.Tensor): Image tensor with shape (1, 3, 32, 32)
        Returns:
            Image: PIL image
        """
        reverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-0.4914/0.2471, -0.4822/0.2435, -0.4465/0.2616],
                                std=[1/0.2471, 1/0.2435, 1/0.2616]),
            lambda x: x.clamp(0, 1),
            transforms.ToPILImage()
        ])
        image = reverse_transform(image_tensor.squeeze())
        
        return image
    
    def predict_image_class(self, image: Image=None, image_tensor: torch.Tensor=None) -> Tuple[str, float]:
        """
        Predicts the class of an image using the ResNet model.
        
        Args:
            image(Image): PIL image
            image_tensor(torch.Tensor): Image tensor with shape (1, 3, 224, 224)
        
        Returns:
            tuple[str, float]: Predicted class and confidence
        """
        if (image is not None) and (image_tensor is None):
            image_tensor = self.preprocess_image(image)
        
        self.model.to(self.device)
        image_tensor = image_tensor.to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class_idx = probabilities.argmax(1).item()
        confidence = probabilities[0, predicted_class_idx].item()
        
        predicted_class = self.classes[predicted_class_idx]
        
        return predicted_class, confidence
    
    def calculate_accuracy(self, folder_path: str, test_dataset_path: str = None) -> float:
        correct_predictions = 0
        total_images = 0

        if test_dataset_path:
            test_files = [f for f in os.listdir(test_dataset_path) if f.endswith('.png')]

        for filename in os.listdir(folder_path):
            if filename.endswith('.png'):
                prefix = filename.split('_')[0]
                if test_dataset_path:
                    image_index = int(prefix)
                    test_filename = test_files[image_index]
                    true_class = test_filename.split('_')[0]
                else:
                    true_class = prefix
                    
                image_path = os.path.join(folder_path, filename)

                image = Image.open(image_path)
                predicted_class, confidence = self.predict_image_class(image=image)

                if predicted_class == true_class:
                    correct_predictions += 1
                total_images += 1

        accuracy = correct_predictions / total_images if total_images > 0 else 0
        return accuracy
    
    def adversarial_attack(self, image: Image, epsilon: float, true_label: int) -> Image:
        """
        Generates an adversarial image using the FGSM method.
        
        Args:
            image(Image): PIL image
            epsilon(float): Attack strength
            true_label(int): True label of the image
        
        Returns:
            Image: Adversarial image
        """
        image_tensor = self.preprocess_image(image).to(self.device)
        image_tensor.requires_grad = True
        
        # Forward pass
        outputs = self.model(image_tensor)
        loss = F.cross_entropy(outputs, torch.tensor([true_label], device=self.device))
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Generate adversarial image
        image_tensor_adv = image_tensor + epsilon * image_tensor.grad.sign()
        
        # Reverse preprocessing
        adversarial_image = self.reverse_preprocess_image(image_tensor_adv)        
        
        return adversarial_image
    
