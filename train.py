import torch
from CNN import ImageClassification
from data import Data

class Trainer():
    def __init__(self, model, epochs, dataloader, device):
        self.model = model.to(device)
        self.epochs = epochs
        self.dataloader = dataloader
        self.device = device


    def train_labeled_data(self):
        model = self.model
        criterion = model.criterion
        optimizer = model.optimizer
        epochs = self.epochs
        dataloader = self.dataloader.labeled_train_loader
        device = self.device
        model.train()

        print("Train labeled data start... ")
        for epoch in range(epochs):
            curr_train_loss = 0
            for idx, (data, label) in enumerate(dataloader):
                data, label = data.to(device), label.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, label)
                curr_train_loss += loss.item()
                loss.backward()
                optimizer.step()

            curr_train_loss /= len(dataloader.dataset)
            print("Epoch: {}, Loss: {}".format((epoch+1), curr_train_loss))


    def train_unlabeled_data(self):
        model = self.model
        criterion = model.criterion
        optimizer = model.optimizer
        epochs = self.epochs
        dataloader = self.dataloader.unlabeled_train_loader
        model.train()

        print("Train unlabeled data start...")
        for epoch in range(epochs):
            # ignore the label
            for idx, (data, _) in enumerate(dataloader):
                optimizer.zero_grad()
                output = model(data)

                # Use the predicted labels as targets
                _, pseudo_lable = torch.max(output, 1)

                loss = criterion(output, pseudo_lable)
                loss.backward()
                optimizer.step()

                    
    def test(self):
        model = self.model
        criterion = model.criterion
        dataloader = self.dataloader.test_loader
        device = self.device

        model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, label in dataloader:
                data, label = data.to(device), label.to(device)

                output = model(data)
                loss = criterion(output, label)
                test_loss += loss.item()
                
                pred = output.argmax(dim=1)
                correct += pred.eq(label).sum()

        test_loss /= len(dataloader.dataset)
        accuracy = float(correct * 100 / len(dataloader.dataset))
        print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(dataloader.dataset), accuracy))


if __name__ == "__main__":
    model = ImageClassification()
    dataloader = Data(5, num_workers=1)
    trainer = Trainer(model, 10, dataloader, torch.device("cpu"))

    trainer.train_labeled_data()
    # trainer.train_unlabeled_data()
    trainer.test()
