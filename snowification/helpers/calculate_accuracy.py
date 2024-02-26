import argparse
from resnet_classifier import ResNetClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', type=str)
parser.add_argument('--model_path', default='./resnet18.pt', type=str)

args = parser.parse_args()


classifier = ResNetClassifier(model_path=args.model_path)

print(f"Calculating accuracy of the classifier on the images in the folder: '{args.folder_path}'", )
print(f"Accuracy: {classifier.calculate_accuracy(folder_path=args.folder_path)}")
