
import torch
from sklearn.metrics import confusion_matrix, f1_score

def image_grid(array, ncols=8):
        index, height, width, channels = array.shape
        nrows = index//ncols

        img_grid = (array.reshape(nrows, ncols, height, width)
                  .swapaxes(1,2)
                  .reshape(height*nrows, width*ncols))

        return img_grid

def get_accuracy(history, testloader, device):

    net = history[4].to(device)
    
    correct = 0
    total = 0
    true_labels = list()
    predicted_labels = list()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            true_labels.extend(labels.cpu())
            predicted_labels.extend(predicted.cpu())
    
    return 100 * correct / total

def get_f1(history, testloader, device):

    net = history[4].to(device)
    
    correct = 0
    total = 0
    true_labels = list()
    predicted_labels = list()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            true_labels.extend(labels.cpu())
            predicted_labels.extend(predicted.cpu())
    
    final_f1 = f1_score(true_labels,predicted_labels, average = 'macro')
    return 100 * final_f1

def get_confusion(history, testloader, device):

    net = history[4].to(device)
    
    correct = 0
    total = 0
    true_labels = list()
    predicted_labels = list()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            true_labels.extend(labels.cpu())
            predicted_labels.extend(predicted.cpu())
    
    cm = confusion_matrix(true_labels, predicted_labels)
    return cm