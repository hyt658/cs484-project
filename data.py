import os
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset

class CIFARData():
    def __init__(self, batch_size, num_workers, label_ratio):
        cifar10_path = "./data/cifar10"
        download = True
        
        # check if dataset has been downloaded
        if os.path.isdir(os.path.join(cifar10_path, "cifar-10-batches-py")):
            download = False

        # Define the transforms for train data augmentation
        train_transform = transforms.Compose([
            transforms.RandomCrop(24, padding=2),
            transforms.RandomRotation(23),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Resize(32, antialias=True)
        ])

        # Define the transforms for test data augmentation
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Resize(32, antialias=True)
        ])

        # obtain the train and test dataset
        train_dataset = datasets.CIFAR10(root=cifar10_path, train=True, download=download, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=cifar10_path, train=False, download=download, transform=test_transform)

        # shuffle the train_dataset to improve generalization
        train_size = len(train_dataset)
        train_indices = np.arange(train_size)
        np.random.shuffle(train_indices)

        # let 10% train data be the labeled, rest are unlabeled
        split_pos = int(train_size * label_ratio)
        labeled_train_indices = train_indices[:split_pos]
        unlabeled_train_indices = train_indices[split_pos:]

        # seperate the train_data set to labeled and unlabeled
        labeled_dataset = Subset(train_dataset, labeled_train_indices)
        unlabeled_dataset = Subset(train_dataset, unlabeled_train_indices)

        labeled_train_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        unlabeled_train_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.labeled_train_loader = labeled_train_loader
        self.unlabeled_train_loader = unlabeled_train_loader
        self.test_loader = test_loader
        self.train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class CombinedDataset(Dataset):
    def __init__(self, labeled_data, unlabeled_data):
        self.labeled_data = labeled_data
        self.unlabeled_data = unlabeled_data

    def __len__(self):
        return len(self.labeled_data) + len(self.unlabeled_data)
    
    def __getitem__(self, idx):
        if idx < len(self.labeled_data):
            return self.labeled_data[idx]
        else:
            data, _ = self.unlabeled_data[idx - len(self.labeled_data)]
            return data, -1


def listToDataloader(data_list, batch_size, num_workers):
    dataset = CustomDataset(data_list)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def indiceSubsetDatalodaer(dataloader, indices):
    batch = dataloader.batch_size
    worker = dataloader.num_workers

    origin_dataset = dataloader.dataset
    new_dataset = Subset(origin_dataset, indices)
    dataloader = DataLoader(new_dataset, shuffle=True, batch_size=batch, num_workers=worker)
    return dataloader


def concatDataloader(data1, data2, has_unlabel=False):
    dataloader = None
    batch = data1.batch_size
    worker = data1.num_workers

    if has_unlabel:
        combined_data = CombinedDataset(data1.dataset, data2.dataset)
        dataloader = DataLoader(combined_data, shuffle=True, batch_size=batch, num_workers=worker)
    else:
        combined_data = ConcatDataset([data1.dataset, data2.dataset])
        dataloader = DataLoader(combined_data, shuffle=True, batch_size=batch, num_workers=worker)

    return dataloader
