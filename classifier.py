import torch
import torch.nn as nn
import torch.nn.functional as F
import opacus                                           
from opacus.validators import ModuleValidator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, f1_score
from config import USE_CELL_DATA, USE_MNIST_DATA, CLASS_NAMES
import numpy as np
import math
from utils import get_f1, get_accuracy, get_confusion

NUM_CLASSES = len(CLASS_NAMES)

class Cellface_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 3)
        self.conv2 = nn.Conv2d(6, 12, 3, 1)
        self.conv3 = nn.Conv2d(12, 18, 3, 1)
        
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(18 * 2 * 2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

class MNIST_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1)
        self.conv2 = nn.Conv2d(6, 12, 3, 1)
        self.conv3 = nn.Conv2d(12, 18, 3, 1)
        
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        
        self.fc1 = nn.Linear(18 * 3 * 3, 96)
        self.fc2 = nn.Linear(96, 10)    

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
    
    
def create_classifier():
    
    if USE_CELL_DATA:
        return Cellface_Net()
    if USE_MNIST_DATA:
        return MNIST_Net()
        
        
def test(net, testloader, device):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

def train(net, trainloader, valloader, epochs, verbose, private, epsilon, delta, device):
    
    train_loss = list()
    train_acc = list()
    val_loss = list()
    val_acc = list()
    
    net = ModuleValidator.fix(net)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
    
    if private:
        privacy_engine = opacus.PrivacyEngine()

        net, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
            module=net,
            optimizer=optimizer,
            data_loader=trainloader,
            target_epsilon=epsilon,
            target_delta=delta,
            epochs = epochs,
            max_grad_norm=2,
        )
    
    for epoch in range(epochs):
        
        correct, total, epoch_loss = 0, 0, 0.0
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            if labels.size(0) == 0: #replace batch with previous one if empty
                labels = lastround_labels 
                images = lastround_images
            lastround_labels = labels
            lastround_images = images
            
            optimizer.zero_grad()
            net.train()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            if private:
                epsilon = privacy_engine.get_epsilon(delta)
                print(f"epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}, ϵ {epsilon}")
            else:
                print(f"epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

        train_loss.append(test(net, trainloader, device)[0])
        train_acc.append(test(net, trainloader, device)[1])
        val_loss.append(test(net, valloader, device)[0])
        val_acc.append(test(net, valloader, device)[1])
        
        
    return train_loss, train_acc, val_loss, val_acc, net

def show_classifier_results(history, testloader, name, device):
    
    train_loss = history[0]
    train_acc = history[1]
    val_loss = history[2]
    val_acc = history[3]
    net = history[4]
    
    
    fig = plt.figure(figsize=(12., 6.))
    plt.subplot(1,2,1)
    plt.rc('font',size = 14)
    plt.rc('legend',fontsize = 11)
    plt.rc('xtick',labelsize = 11)
    plt.rc('ytick',labelsize = 11)
    plt.title('Loss history')# of '+ name)
    plt.plot(train_loss,linestyle="-", label = "training",color="#a8e6cf")
    plt.plot(val_loss,linestyle="-", label = "validation",color="#ff8b94")
    plt.plot(val_loss.index(min(val_loss)), min(val_loss), 'yo',label = f'validation minimum at epoch {val_loss.index(min(val_loss))+1}')
    plt.xticks(np.arange(len(train_loss)),np.arange(1,len(train_loss)+1))
    plt.xticks(np.arange(1,len(train_loss)+1,step = math.ceil(len(train_loss)/5)))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    #plt.ylim([0,0.06])
    plt.grid(axis='y', color='0.8')
    plt.gca().set_frame_on(False)
    plt.legend()



    plt.subplot(1,2,2)
    plt.rc('font',size = 14)
    plt.rc('legend',fontsize = 11)
    plt.rc('xtick',labelsize = 11)
    plt.rc('ytick',labelsize = 11)
    plt.title('Accuracy history')# of '+ name)
    plt.plot(train_acc,linestyle="-", label = "training",color="#a8e6cf")
    plt.plot(val_acc,linestyle="-", label = "validation",color="#ff8b94")
    plt.xticks(np.arange(len(train_acc)),np.arange(1,len(train_acc)+1))
    plt.xticks(np.arange(1,len(train_acc)+1,step = math.ceil(len(train_acc)/5)))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    #plt.ylim([0,1])
    plt.grid(axis='y', color='0.8')
    plt.gca().set_frame_on(False)
    plt.legend()
    plt.show()
    #plt.savefig(name)

    accuracy = get_accuracy(history, testloader, device)
    f1 = get_f1(history, testloader, device)
    cm = get_confusion(history, testloader, device)
    
    #plt.title('Confusion matrix of '+ name)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig = plt.figure(figsize=(10., 10.))
    disp.plot()
    print('Accuracy of ' + name + f' on the test images: {accuracy} %')
    print('F1 of ' + name + f' on the test images: {f1} %')
    
