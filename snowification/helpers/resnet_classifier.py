from typing import Optional, Union
import torch
import torch.nn.functional as F
import torchvision.models as models

from PIL import Image
from torch import nn
from torchvision import transforms

class ResNetClassifier:        
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, model_path: str):
        self.model = self._load_resnet_model(model_path=model_path)

    def _load_resnet_model(self, model_path: str) -> models.ResNet:
        resnet = models.resnet18(pretrained=False)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, 10)
        
        state_dict = torch.load(model_path, map_location=self.device)
        resnet.load_state_dict(state_dict)
        model = resnet.to(self.device)
        
        return model
    
    def preprocess_image(self, image: Image) -> torch.Tensor:
        """
        Preprocesses an image to be used as input for the ResNet model.
        
        Args:
            image(Image): PIL image
        Returns:
            torch.Tensor: Image tensor with shape (1, 3, 224, 224)
        """
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = preprocess(image).unsqueeze(0)
        
        return image_tensor
    
    def reverse_preprocess_image(self, image_tensor: torch.Tensor) -> Image:
        """
        Reverses the preprocessing of an image.
        
        Args:
            image_tensor(torch.Tensor): Image tensor with shape (1, 3, 224, 224)
        Returns:
            Image: PIL image (32 x 32)
        """
        reverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                 std=[1/0.229, 1/0.224, 1/0.225]),
            lambda x: x.clamp(0, 1),
            transforms.ToPILImage(),
            transforms.Resize((32, 32))
        ])
        image = reverse_transform(image_tensor.squeeze())
        
        return image
    
    def predict_image_class(self, image: Image=None, image_tensor: torch.Tensor=None) -> tuple[str, float]:
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
    