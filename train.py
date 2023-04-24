import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data import CIFARData, listToDataloader
from CNN import ImageClassifier
from collections import defaultdict
from copy import deepcopy
from torch.utils.data import DataLoader, ConcatDataset
from data import CustomDataset

def getDeviceName():
    if (torch.cuda.is_available()):
        return "cuda"
    elif (torch.backends.mps.is_available()):
        return "mps"
    else:
        return "cpu"


def getBestTeacher(is_train, label_data, test_data):
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


def train(model, dataloader, criterion, optimizer, epochs=20):
    model.to(DEVICE)
    model.train()

    for epoch in range(epochs):
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


def predict(model, dataloader, device, batch_size, threshold=0.9):
    model.to(device)
    model.eval()
    msg_len = 0

    # result will be a list of tuple (data_idx, label)
    result = []

    # sample_data is a dict of list, format is
    #   label: [data_idx, probability of this data belongs to this label]
    sample_data = defaultdict(list)

    # predict the label by given model, for each data use its topk possible
    #   label, then put them into sample_data
    print("Prediction starts...")
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)

            output = model(data)
            softmax_output = F.softmax(output, dim=-1)
            topk, topk_idx = softmax_output.topk(k=10, dim=1)

            data = data.cpu()
            topk = topk.cpu()
            topk_idx = topk_idx.cpu()

            for idx in range(batch_size):
                curr = data[idx]
                for i, label in enumerate(topk_idx[idx]):
                    if topk[idx][i] > threshold:
                        sample_data[label].append((batch_idx * batch_size + idx, topk[idx][i]))

            if (batch_idx + 1) % 10 == 0:
                print(' ' * msg_len, end='\r')
                databatch_count = len(dataloader)
                percent = float(batch_idx / databatch_count * 100)
                msg = "Progress: {}/{} ({:.1f}%)".format(batch_idx + 1, databatch_count, percent)
                print(msg, end='\r')
                msg_len = len(msg)

    print(' ' * msg_len, end='\r')
    print("Packing prediction result...")
    indices_list = []
    for label in sample_data:
        sort_func = lambda x: x[1]  # x[1] is the probability
        chosen = sorted(sample_data[label], key=sort_func)
        for data_prob_pair in chosen:
            indices_list.append(data_prob_pair[0])

    print(len(indices_list))

    # Create a DataLoader from the result list
    predicted_dataset = CustomDataset(dataloader.dataset, indices_list)
    predicted_dataloader = DataLoader(predicted_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return predicted_dataloader

                    
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
    EPOCHS = 20
    BATCH = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_SAVE_PATH = "./saved_model"
    THRESHOLD = 0.8
    ACC_THRESHOLD = 0.01
    STOP_THRESHOLD = 3

    dataloader = CIFARData(batch_size=BATCH, num_workers=4, unlabel_ratio=0.1)
    label_data = dataloader.labeled_train_loader
    unlabeled_data = dataloader.unlabeled_train_loader
    test_data = dataloader.test_loader

    # get teacher model
    teacher = getBestTeacher(is_train=True, label_data=label_data, test_data=test_data)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher.parameters(), lr=0.001)

    # get teacher accuracy
    teacher_test_loss, teacher_test_accuracy = test(teacher, test_data, criterion)

    # 打印教师模型准确度
    print(f"Teacher accuracy on test set: {teacher_test_accuracy:.4f}")

    # 初始化循环条件和计数器
    improved = True
    stop_count = 0
    count = 0

    # Initialize the best accuracy and the best model
    best_accuracy = 0
    best_model = None

    while improved:
        # Use teacher model to predict unlabelled data applying the threshold
        print("Teacher starts to predict unlabeled data with threshold...")
        prediction = predict(teacher, unlabeled_data, device="cuda", batch_size=BATCH, threshold=THRESHOLD)

        combined_data = torch.utils.data.ConcatDataset([label_data.dataset, prediction.dataset])
        combined_data_loader = torch.utils.data.DataLoader(combined_data, batch_size=BATCH, shuffle=True, num_workers=4)

        # use labelled data and (fake) new labelled data together to train a new student model
        print("Student starts to be trained with prediction data...")
        student = ImageClassifier()
        optimizer = optim.Adam(student.parameters(), lr=0.001)
        student_train_loss = train(student, combined_data_loader, criterion, optimizer, epochs=20)
        student_test_loss, student_test_accuracy = test(student, test_data, criterion)

        print("student accuracy on test:", student_test_accuracy)
        print("teacher accuracy on test:", teacher_test_accuracy)

        # accuracy improvement calculation
        accuracy_improvement = student_test_accuracy - teacher_test_accuracy

        # if student is worse than teacher, stop_count += 1
        if accuracy_improvement < ACC_THRESHOLD:
            stop_count += 1
        else:
            stop_count = 0

        # if the performance  of the model has not been improved for several times, stop
        if stop_count == STOP_THRESHOLD:
            improved = False
        elif accuracy_improvement > ACC_THRESHOLD:
            # Update the best accuracy and the best model
            if student_test_accuracy > best_accuracy:
                best_accuracy = student_test_accuracy
                best_model = deepcopy(student)

            # update teacher model to the better student model
            teacher = deepcopy(student)
            teacher_test_accuracy = student_test_accuracy
            label_data = combined_data_loader

        # count increment
        count += 1

        if count == EPOCHS:
            improved = False

    print(f"Final student accuracy on test set: {student_test_accuracy:.4f}")

    # Save the best model
    torch.save(best_model.state_dict(), os.path.join(MODEL_SAVE_PATH, "best_model.pth"))
    print(f"Best student accuracy on test set: {student_test_accuracy:.4f}")

