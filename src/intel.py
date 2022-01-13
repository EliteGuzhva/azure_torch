# imports
import torchvision
import torch.nn as nn
from torch import optim
from torchvision import transforms, models
import torchvision.transforms as transforms

from train import *


# define network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.model.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# run script
if __name__ == "__main__":
    # setup argparse
    parser = get_parser()
    args = parser.parse_args()

    # define train and test dataset DataLoaders
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.3, 0.4, 0.4, 0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.425, 0.415, 0.405), (0.205, 0.205, 0.205))
    ])

    train_data_dir = args.data_dir + "/seg_train/seg_train"
    train_set = torchvision.datasets.ImageFolder(root=train_data_dir, transform=transform)

    transform_test = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_data_dir = args.data_dir + "/seg_test/seg_test"
    test_set = torchvision.datasets.ImageFolder(root=test_data_dir, transform=transform_test)

    # define loss function and optimizer
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.learning_rate)

    # call main function
    main(args, model, criterion, optimizer, train_set, test_set)

