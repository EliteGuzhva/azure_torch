# imports
import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


# define functions
def get_parser():
    # setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        help="directory containing dataset"
    )
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        help="number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        default=16,
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
        "--learning-rate",
        default=0.001,
        type=float,
        help="learning rate"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        type=str,
        help="directory to save model to"
    )
    parser.add_argument(
        "--print-freq",
        default=10,
        type=int,
        help="frequency of printing training statistics",
    )

    return parser


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


def main(args, model, criterion, optimizer, train_set, test_set):
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

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=True
    )

    model = model.to(device)

    # wrap model with DDP
    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)

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
            "learning_rate": args.learning_rate,
            "num_nodes": world_size
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

