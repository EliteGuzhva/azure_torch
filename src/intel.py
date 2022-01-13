# imports
import os
import yaml
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.distributed as dist
import torchvision.transforms as transforms
from torch import optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


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


# define functions
def train(train_loader, model, criterion, optimizer, epoch, device, print_freq, rank):
    ret_loss = 0.0
    ret_accuracy = 0.0
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()
        if i > 0 and i % print_freq == 0:  # print every print_freq mini-batches
            cur_loss = running_loss / print_freq
            cur_accuracy = 100 * correct / total
            print(f"Rank {rank}: [{epoch + 1}, {i}] "
                  f"loss: {cur_loss:.3f} "
                  f"accuracy: {cur_accuracy:.3f}%")
            ret_loss = cur_loss
            ret_accuracy = cur_accuracy
            running_loss = 0.0
            correct = 0
            total = 0
    
    return ret_loss, ret_accuracy


def evaluate(test_loader, model, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # print total test set accuracy
    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy}%")


def main(args):
    # get PyTorch environment variables
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    distributed = world_size > 1

    # set device
    if distributed:
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize distributed process group using default env:// method
    if distributed:
        dist.init_process_group(backend="nccl")

    # define train and test dataset DataLoaders
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.3, 0.4, 0.4, 0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.425, 0.415, 0.405), (0.205, 0.205, 0.205))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_data_dir = args.data_dir + "/seg_train/seg_train"
    train_set = torchvision.datasets.ImageFolder(root=train_data_dir, transform=transform)

    if distributed:
        train_sampler = DistributedSampler(train_set)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        sampler=train_sampler,
        drop_last=True
    )

    test_data_dir = args.data_dir + "/seg_test/seg_test"
    test_set = torchvision.datasets.ImageFolder(root=test_data_dir, transform=transform_test)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=True
    )

    model = Net().to(device)

    # wrap model with DDP
    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.learning_rate)

    # train the model
    losses = []
    accuracies = []
    for epoch in range(args.epochs):
        print(f"Rank {rank}: Starting epoch {epoch + 1}")
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        loss, accuracy = train(train_loader, model, criterion,
                               optimizer, epoch, device, args.print_freq, rank)
        losses.append(loss)
        accuracies.append(accuracy)

    print(f"Rank {rank}: Finished Training")

    if not distributed or rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

        model_path = os.path.join(args.output_dir, "model.pt")
        torch.save(model.state_dict(), model_path)

        info = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate
        }
        info_path = os.path.join(args.output_dir, "info.yaml")
        with open(info_path, 'w') as f:
            yaml.safe_dump(info, f)

        loss_path = os.path.join(args.output_dir, "log.csv")
        with open(loss_path, 'w') as f:
            f.write("epoch,loss,accuracy\n")
            for i in range(len(losses)):
                f.write(f"{i+1},{losses[i]},{accuracies[i]}\n")

        # evaluate on full test dataset
        evaluate(test_loader, model, device)


# run script
if __name__ == "__main__":
    # setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=str, help="directory containing Intel image classification dataset"
    )
    parser.add_argument("--epochs", default=10, type=int, help="number of epochs")
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help="mini batch size for each gpu/process",
    )
    parser.add_argument(
        "--workers",
        default=2,
        type=int,
        help="number of data loading workers for each gpu/process",
    )
    parser.add_argument(
        "--learning-rate", default=1e-3, type=float, help="learning rate"
    )
    parser.add_argument(
        "--output-dir", default="outputs", type=str, help="directory to save model to"
    )
    parser.add_argument(
        "--print-freq",
        default=50,
        type=int,
        help="frequency of printing training statistics",
    )
    args = parser.parse_args()

    # call main function
    main(args)


