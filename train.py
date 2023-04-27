import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data import *
from CNN import *
from copy import deepcopy
from collections import defaultdict

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def getBestTeacher(is_train, label_data, test_data):
    teacher = ResNet(num_blocks=[2, 2, 2, 2])
    train_loss = 0
    test_loss = 0
    test_accuracy = 0
    teacher_path = os.path.join(MODEL_SAVE_PATH, "best_teacher_model_{}.pt".format(DEVICE))
    find_save = os.path.exists(teacher_path)

    if find_save:
        # load from saved model
        checkpoint = torch.load(teacher_path)

    if not is_train:
        # check if we have model saved
        if not find_save:
            print("Saved teacher model is not detected, start training a new one...")
            is_train = True
        
    if is_train:
        # train teacher model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(teacher.parameters(), lr=0.001)
        print("Teacher starts to be trained with labeled data...")
        train_loss = train(teacher, label_data, criterion, optimizer)
        test_loss, test_accuracy = test(teacher, test_data, criterion)

        state = {
            "model": teacher.state_dict(), 
            "train_loss": train_loss,
            "test_loss": test_loss,
            "accuracy": test_accuracy
        }

        if find_save:
            # if this train result is better, we update the old checkpoint
            saved_accuracy = checkpoint["accuracy"]
            if (test_accuracy > saved_accuracy):
                torch.save(state, teacher_path)
        else:
            # if no checkpoint, save one
            torch.save(state, teacher_path)
    else:
        print("Load saved Teacher model...")
        teacher.load_state_dict(checkpoint["model"])
    
    return teacher, test_accuracy


def train(model, dataloader, criterion, optimizer):
    model.to(DEVICE)
    model.train()

    for epoch in range(EPOCHES):
        best_train_loss = 0
        curr_train_loss = 0
        msg_len = 0

        for batch_idx, (data, label) in enumerate(dataloader):
            data, label = data.to(DEVICE), label.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            curr_train_loss += loss.item()
            loss.backward()
            optimizer.step()

            # print out the progress (not suitable for notebook)
            print(' ' * msg_len, end='\r')      # clean current line
            databatch_count = len(dataloader)
            percent = float(batch_idx / databatch_count * 100)
            msg = "Epoch: {}, Progress: {}/{} ({:.1f}%)".format(
                epoch+1, batch_idx+1, databatch_count, percent)
            print(msg, end='\r')
            msg_len = len(msg)

        print(' ' * msg_len, end='\r')
        data_count = len(dataloader.dataset)
        curr_train_loss /= data_count
        print("Epoch: {}, Loss: {}".format((epoch+1), curr_train_loss))

        if (curr_train_loss < best_train_loss):
            best_train_loss = curr_train_loss

    return best_train_loss


def predict(model, dataloader, threshold):
    model.to(DEVICE)
    model.eval()
    msg_len = 0

    # result will be list of tuple (data, label)
    result = []

    # the indices of not predicting properly data in dataset
    unused_indices = []

    # sample_data is a dict of list, format is
    #   label: [data_idx, probability of this data belongs to this label]
    sample_data = defaultdict(list)

    # predict the label by given model, for each data use its topk possible
    #   label, then put them into sample_data
    print("Prediction starts...")
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(DEVICE)

            output = model(data)
            softmax_output = F.softmax(output, dim=-1)
            topk, topk_idx = softmax_output.topk(k=10, dim=1)

            data = data.cpu()
            topk = topk.cpu()
            topk_idx = topk_idx.cpu()

            for idx in range(topk_idx.size(0)):
                is_good_predict = False
                for i, label in enumerate(topk_idx[idx]):
                    if topk[idx][i] > threshold:
                        curr = data[idx]
                        sample_data[label.item()].append((curr, topk[idx][i]))
                        is_good_predict = True
                if not is_good_predict:
                    curr_idx = batch_idx * BATCHES + idx
                    unused_indices.append(curr_idx)

            # print out the progress (not suitable for notebook)
            if ((batch_idx+1) % 10 == 0):
                print(' ' * msg_len, end='\r')      # clean current line
                databatch_count = len(dataloader)
                percent = float(batch_idx / databatch_count * 100)
                msg = "Progress: {}/{} ({:.1f}%)".format(batch_idx+1, databatch_count, percent)
                print(msg, end='\r')
                msg_len = len(msg)
    
    # now base on sample_data, for each label, let the top 250 data (sort in probability) that owns this label
    print("\nPacking prediction result...", end='')
    for label in sample_data:
        sort_func = lambda x: x[1]  # x[1] is the probability
        chosen = sorted(sample_data[label], key=sort_func, reverse=True)[:250]
        for data_prob_pair in chosen:
            result.append((data_prob_pair[0], label))
    print("Done\n")

    predict = listToDataloader(result, BATCHES, WORKERS)

    return predict, unused_indices

                    
def test(model, dataloader, criterion):
    model.to(DEVICE)
    model.eval()
    test_loss = 0
    correct = 0
    
    print("Testing starts...")
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(DEVICE), label.to(DEVICE)

            output = model(data)
            loss = criterion(output, label)
            test_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(label).sum()

    data_count = len(dataloader.dataset)
    test_loss /= data_count
    accuracy = float(correct / data_count * 100)
    print("Test result - Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n".format(
        test_loss, correct, data_count, accuracy))
    
    return test_loss, accuracy


def train_teacher_student(label_ratio, save_student=True, identify=""):
    threshold = 0.9
    acc_threshold = 0.3
    stop_threshold = 3

    dataloader = CIFARData(batch_size=BATCHES, num_workers=WORKERS, label_ratio=label_ratio)
    label_data = dataloader.labeled_train_loader
    unlabeled_data = dataloader.unlabeled_train_loader
    test_data = dataloader.test_loader

    # get teacher model
    teacher, teacher_test_accuracy = getBestTeacher(is_train=True, label_data=label_data, test_data=test_data)

    # initial variables for training loop
    improved = True
    stop_count = 0
    count = 0

    # collect the best student model
    best_model = None
    best_train_loss = torch.inf
    best_test_loss = torch.inf
    best_accuracy = 0

    # the indices of data in unlabeled data that didn't get well predicted
    unused_indices = []

    while improved:
        # use teacher to predict unlabeled data, then take good predicted data out
        print("Teacher starts to predict unlabeled data...")
        pred_data, unused_indices = predict(teacher, unlabeled_data, threshold)

        # combined labeled and prediction data
        combined_data = concatDataloader(label_data, pred_data)

        # train the student with prediction data
        student = ResNet(BasicBlock, [2, 2, 2, 2])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(student.parameters(), lr=0.0005)
        print("Student starts to be trained with prediction data...")
        student_train_loss = train(student, combined_data, criterion, optimizer)
        student_test_loss, student_test_accuracy = test(student, test_data, criterion)

        # update the best student and model info
        if student_test_accuracy > best_accuracy:
            best_model = deepcopy(student)
            best_train_loss = student_train_loss
            best_test_loss = student_test_loss
            best_accuracy = student_test_accuracy

        # accuracy improvement calculation
        accuracy_improvement = student_test_accuracy - teacher_test_accuracy

        # if student is worse than teacher, we increase stop_count
        if accuracy_improvement < acc_threshold:
            stop_count += 1
        else:
            stop_count = 0

        # if the performance  of the model has not been improved for several times, stop
        if stop_count == stop_threshold:
            improved = False
        elif accuracy_improvement > acc_threshold:
            # update teacher model to the better student model
            teacher = deepcopy(student)
            teacher_test_accuracy = student_test_accuracy

            # since we use the new student as teacher, we move its good predicted unlabeled
            #   data to labled
            unlabeled_data = indiceSubsetDatalodaer(unlabeled_data, unused_indices)
            label_data = combined_data

        count += 1
        threshold -= 0.005

        if count == EPOCHES:
            improved = False
        
    print("the final size of label_data is {}".format((len(label_data.dataset))))
    print("Best student accuracy is {}%".format(best_accuracy))

    if save_student:
        student_path = os.path.join(MODEL_SAVE_PATH, "best_student_model_{}_{}.pt".format(DEVICE, identify))
        find_save = os.path.exists(student_path)
        state = {
            "model": best_model.state_dict(), 
            "train_loss": best_train_loss,
            "test_loss": best_test_loss,
            "accuracy": best_accuracy
        }

        # check if we have student saved. if does, compare the accuracy and save the best one
        if (find_save):
            checkpoint = torch.load(student_path)
            saved_acc = checkpoint["accuracy"]
            if (best_accuracy > saved_acc):
                torch.save(state, student_path)
        else:
            torch.save(state, student_path)
    

def kmeans_loss(features, centroids):
    distances = torch.cdist(features, centroids)
    min_distances, _ = torch.min(distances, dim=1)
    loss = torch.mean(min_distances ** 2)
    return loss


def train_combined(model, dataloader, optimizer, criterion, device, kmeans_weight, num_classes):
    model.train()
    total_loss = 0.0
    sup_loss = 0

    for batch_idx, (data, targets) in enumerate(dataloader):
        data, targets = data.to(device), targets.to(device)
        labeled_mask = targets != -1

        optimizer.zero_grad()
        
        # Get the output before the linear classifier
        features = model.features(data)
        
        # Compute the K-means clustering loss
        centroids = torch.randn(num_classes, features.size(1), device=device)
        # centroids = initialize_centroids(features, num_classes)
        kmeans_loss_value = kmeans_loss(features, centroids)
        
        # Compute the cross-entropy loss for labeled data
        output = model.classifier(features)
        ce_loss_value = 0
        if labeled_mask.sum() > 0:
            ce_loss_value = criterion(output[labeled_mask], targets[labeled_mask])
        
        # Combine the losses
        loss = ce_loss_value + kmeans_weight * kmeans_loss_value
        # print("ce_loss_value: ", ce_loss_value.item())
        # print("kmeans_loss", kmeans_weight * kmeans_loss_value.item())
        loss.backward()
        
        optimizer.step()
        
        total_loss += loss.item()
        sup_loss += ce_loss_value

    return total_loss / (len(dataloader.dataset)), (total_loss - sup_loss) / (len(dataloader.dataset))


def training_combined_complete(label_ratio, save_model=True, identify=""):
    stop_threshold = 3

    dataloader = CIFARData(batch_size=BATCHES, num_workers=WORKERS, label_ratio=label_ratio)
    label_data = dataloader.labeled_train_loader
    unlabeled_data = dataloader.unlabeled_train_loader
    test_data = dataloader.test_loader

    # get supervised model
    sup_resnet, _ = getBestTeacher(True, label_data, test_data)

    # initial variables for training loop
    stop_count = 0
    count = 0

    # collect the best accuracy and model
    best_model = None
    best_train_loss = torch.inf
    best_test_loss = torch.inf
    best_accuracy = 0

    combined_data = CombinedDataset(label_data, unlabeled_data)
    combined_dataloader = torch.utils.data.DataLoader(combined_data, batch_size=BATCHES, shuffle=True, num_workers=WORKERS)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(sup_resnet.parameters(), lr=0.001)

    for epoch in range(1, EPOCHES + 1):
        train_loss, km_loss = train_combined(sup_resnet, combined_dataloader, optimizer, criterion, DEVICE, kmeans_weight=0.005, num_classes=10)

        print("kmeans loss: ", km_loss.item())
        print("mean total loss", train_loss)
        test_loss, accuracy = test(sup_resnet, test_data, criterion)
      

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = deepcopy(sup_resnet)
            stop_count = 0
        else:
            stop_count += 1

        if stop_count >= stop_threshold:
            break

    print("Best accuracy is {}%".format(best_accuracy))

    if save_model:
        if identify == "":
          identify = str(label_ratio)

        sup_kmeans_path = os.path.join(MODEL_SAVE_PATH, "best_sup_kmeans_model_{},pt".format(identify))
        find_save = os.path.exists(sup_kmeans_path)
        state = {
            "model": best_model.state_dict(), 
            "train_loss": best_train_loss,
            "test_loss": best_test_loss,
            "accuracy": best_accuracy
        }

        # check if we have model saved. if does, compare the accuracy and save the best one
        if (find_save):
            checkpoint = torch.load(sup_kmeans_path)
            saved_acc = checkpoint["accuracy"]
            if (best_accuracy > saved_acc):
                torch.save(state, sup_kmeans_path)
        else:
            torch.save(state, sup_kmeans_path)


if __name__ == "__main__":
    EPOCHES = 20
    BATCHES = 100
    WORKERS = 10
    MODEL_SAVE_PATH = "./saved_model"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # train_teacher_student(0.3, True, "30%")
    # training_combined_complete(0.3, True, "30%")

    path = os.path.join(MODEL_SAVE_PATH, "best_student_model_cuda_30%.pt")
    checkpoint = torch.load(path)
    model = ResNet([2, 2, 2, 2])
    model.load_state_dict(checkpoint["model"])
 
    img = Image.open("./image/img4.jpg")
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(32, antialias=True)
    ])
    tensor = test_transform(img)

    cifar10 = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transforms.ToTensor())
    image, label = cifar10[888]
    print(label)
    output = model(image.unsqueeze(0))

    # output = model(tensor.unsqueeze(0))
    probs, predicted = torch.max(output.data, 1)
    print(predicted.item())
