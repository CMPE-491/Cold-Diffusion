import argparse
from resnet_classifier import ResNetClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', type=str)
parser.add_argument('--model_path', default='./cifar10_resnet18.pth', type=str)

args = parser.parse_args()


classifier = ResNetClassifier(model_path=args.model_path)

print(classifier.calculate_accuracy(folder_path=args.folder_path))
