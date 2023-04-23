import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data import CIFARData, listToDataloader
from CNN import ImageClassifier
from collections import defaultdict


def getBestTeacher(is_train):
    teacher = ImageClassifier()
    train_loss = 0
    test_loss = 0
    test_accuracy = 0
    teacher_path = os.path.join(MODEL_SAVE_PATH, "best_teacher_model.pt")
    find_save = os.path.exists(teacher_path)

    if find_save:
        # load from saved model
        checkpoint = torch.load(os.path.join(MODEL_SAVE_PATH, "best_teacher_model.pt"))

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
    
    return teacher


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

            if ((batch_idx+1) % 10 == 0):
                print(' ' * msg_len, end='\r')      # clean current line
                databatch_count = len(dataloader)
                percent = float(batch_idx / databatch_count * 100)
                msg = "Epoch: {}, Progress: {}/{} ({:.1f}%)".format(
                    epoch+1, batch_idx+1, databatch_count, percent)
                print(msg, end='\r')
                msg_len = len(msg)

        data_count = len(dataloader.dataset)
        curr_train_loss /= data_count
        print(' ' * msg_len, end='\r')
        print("Epoch: {}, Loss: {}".format((epoch+1), curr_train_loss))

        if (curr_train_loss < best_train_loss):
            best_train_loss = curr_train_loss

    return best_train_loss


def predict(model, dataloader):
    model.to(DEVICE)
    model.eval()
    msg_len = 0

    # result will be list of tuple (data, label)
    result = []

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

            for idx in range(BATCH):
                curr = data[idx]
                for i, label in enumerate(topk_idx[idx]):
                    sample_data[label].append((curr, topk[idx][i]))

            if ((batch_idx+1) % 10 == 0):
                # clean current line
                print(' ' * msg_len, end='\r')
                databatch_count = len(dataloader)
                percent = float(batch_idx / databatch_count * 100)
                msg = "Progress: {}/{} ({:.1f}%)".format(batch_idx+1, databatch_count, percent)
                print(msg, end='\r')
                msg_len = len(msg)
    
    # now base on sample_data, for each label, let the first 100
    #   data (sort in probability) that own this label
    print(' ' * msg_len, end='\r')
    print("Packing prediction result...")
    for label in sample_data:
        sort_func = lambda x: x[1]  # x[1] is the probability
        chosen = sorted(sample_data[label], key=sort_func)[:100]
        for data_prob_pair in chosen:
            result.append((data_prob_pair[0], label))

    return result

                    
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


if __name__ == "__main__":
    EPOCHES = 20
    BATCH = 10
    DEVICE = torch.device("mps")
    MODEL_SAVE_PATH = "./saved_model"

    dataloader = CIFARData(batch_size=BATCH, num_workers=1)
    label_data = dataloader.labeled_train_loader
    unlabeled_data = dataloader.unlabeled_train_loader
    test_data = dataloader.test_loader

    # get a teacher model (either train or load checkpoint)
    teacher = getBestTeacher(is_train=False)

    # use teacher to predict unlabeled data
    print("Teacher starts to predict unlabeled data...")
    prediction = predict(teacher, unlabeled_data)
    pred_data = listToDataloader(prediction, batch_size=BATCH, num_workers=1)

    # train the student with prediction data
    student = ImageClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    print("Student starts to be trained with prediction data...")
    student_train_loss = train(student, pred_data, criterion, optimizer)
    student_test_loss, student_test_accuracy = test(student, test_data, criterion)

    # finally tune student with labeled data
    print("Student starts to be tuned with labeled data...")
    student_train_loss = train(student, label_data, criterion, optimizer)
    student_test_loss, student_test_accuracy = test(student, test_data, criterion)
