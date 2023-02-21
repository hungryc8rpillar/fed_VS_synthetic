
import torch
from torchvision import transforms
import numpy as np
from cellface import * 
from cellface.storage.container import *
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import Subset

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


def read_data(class_names, class_paths, sample_limit):
    if len(class_names) != len(class_paths):
        print('Error: Number of classes unequal to numer of paths')
    else:
        sets = dict()
        for i in range(len(class_names)):
            sets[class_names[i]] = read_class(class_paths[i], sample_limit)
            
        return sets
    
def read_class(seg_path, sample_limit):
    
    with Seg(seg_path, "r") as seg:
    
        image = seg.content.phase.images[:sample_limit]
        masks = seg.content.patch_specification.masks[:sample_limit]
        maskedim = image[()].compute() * masks['mask'][()].compute()
        
    return maskedim

    
def process_sets(sets,balance, img_size):
    
    #TODO: shuffle
    #TODO: control over split
    #TODO progbar
    
    #balance for shortest set
    shortest_set_length = min([len(entry) for entry in sets.values()])

    for k in sets:
        print('Processing now ' + str(k))
        if balance:
            sets[k] = sets[k][:shortest_set_length]
            
        sets[k] = torch.tensor(np.nan_to_num(sets[k]))
        
        for i in range(len(sets[k])):
            sets[k][i] = process_image(sets[k][i], img_size)
    return sets
            

def process_image(image, img_size):
    
    #resize to IMG_SIZE
    image = transforms.functional.resize(image[None, :], size = img_size)[0,:]
    image = np.array(image)
    
    
    #remove segmentation masks errors
    column_index = []
    row_index = []
    
    for i in range(len(image)):
        if 0 not in image[:,i]:
            column_index.append(i)
        if 0 not in image[i,:]:
            row_index.append(i)

    for index in column_index:
        image[:,index] = np.zeros(len(image))
    for index in row_index:  
        image[index,:] = np.zeros(len(image))
    
    
    #scale to [-1,1] for neural net (generator outputs tanh)
    imax = np.max(np.array(image))
    imin = np.min(np.array(image))
    
    for i in range(len(image)):
        image[i] = 2 * (image[i] - imin) / (imax - imin + 0.00001) - 1
        
    return torch.tensor(image)

def make_torch_sets(sets):
    target = 0
    for k in sets:
        sets[k] = Dataset_Class(sets[k], target)
        target += 1
        
    #return sets["RBC"]
    final_set = torch.utils.data.ConcatDataset(list(sets.values()))
    
    return final_set
    

class Dataset_Class(Dataset):
        
        def __init__(self, input_data, target):
            self.data = np.array(input_data)
            self.target = target
            
        def __getitem__(self, idx):
            sample = self.data[idx]
            data, label = sample, self.target
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            return transform(data), torch.tensor(label)
        
        def __len__(self):
            return len(self.data)

def adjust_trainset(dataset, num_partitions): #make trainingsset dividable by partition number
    for i in reversed(range(len(dataset)+1)):
        if i%num_partitions == 0:
            return torch.utils.data.Subset(dataset,list(range(0,i)))
        
def load_datasets(num_partitions, class_names, train_dataset, test_dataset, batch_size_classifier, batch_size_gan):
    
    trainset = train_dataset
    testset = test_dataset
    
    partition_size = len(trainset) // num_partitions #custom split needed here
    lengths = [partition_size] * num_partitions
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))
    len_val = len(train_dataset) // 10  
    len_train = len(train_dataset) - len_val
    lengths = [len_train, len_val]
    
    benchmark_trainset, benchmark_valset = random_split(train_dataset, lengths, torch.Generator().manual_seed(42))
    benchmark_trainloader = DataLoader(benchmark_trainset, batch_size=batch_size_classifier, shuffle=True)
    benchmark_valloader = DataLoader(benchmark_valset, batch_size=batch_size_classifier, shuffle=True)
    
    fed_trainloaders = []
    fed_valloaders = []
    
    gan_trainloaders = []

    for ds in datasets:
        
        all_indices = list()
        for i in range(len(ds)):
            all_indices.append(ds[i][1])
        
        for i in range(len(class_names)):
            mask = torch.tensor(all_indices) == i
            train_indices = mask.nonzero().reshape(-1)
            train_class = Subset(ds, train_indices)
            gan_trainloaders.append(DataLoader(train_class, batch_size=batch_size_gan, shuffle=True, drop_last=True))
            
        len_val = len(ds) // 10  
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        fed_trainloaders.append(DataLoader(ds_train, batch_size=batch_size_classifier, shuffle=True))
        fed_valloaders.append(DataLoader(ds_val, batch_size=batch_size_classifier, shuffle=True))
        
    testloader = DataLoader(testset, batch_size=batch_size_classifier)
    return gan_trainloaders, fed_trainloaders, fed_valloaders, testloader, benchmark_trainloader, benchmark_valloader, train_dataset

def load_noniid_datasets(num_partitions, class_names, train_dataset, test_dataset, batch_size_classifier, batch_size_gan):
    
    trainset = train_dataset
    testset = test_dataset
    all_labels = [int(x[1]) for x in trainset]
    unique_labels = set(all_labels)

    labelsets = dict()
    
    for label in unique_labels:
        
        mask = torch.tensor(all_labels) == label
        train_indices = mask.nonzero().reshape(-1)
        
        splitpoints = np.sort(np.random.uniform(0, 1, num_partitions - 1))*np.sum(np.array(mask))
        splitpoints = np.append(0,splitpoints)
        splitpoints = np.append(splitpoints,int(np.sum(np.array(mask))))
        splitpoints = splitpoints.astype(int)
                
        lengths = list()
        for i in range(len(splitpoints)-1):
            lengths.append(splitpoints[i+1]-splitpoints[i])
        labelsets[label] = random_split(Subset(trainset, train_indices), lengths, torch.Generator().manual_seed(42))

    datasets = list()
    for i in range(num_partitions):
        partition = list()
        for label in unique_labels:
            partition.append(labelsets[label][i])
        datasets.append(torch.utils.data.ConcatDataset(partition))

    len_val = len(train_dataset) // 10  
    len_train = len(train_dataset) - len_val
    lengths = [len_train, len_val]
    
    benchmark_trainset, benchmark_valset = random_split(train_dataset, lengths, torch.Generator().manual_seed(42))
    benchmark_trainloader = DataLoader(benchmark_trainset, batch_size=batch_size_classifier, shuffle=True)
    benchmark_valloader = DataLoader(benchmark_valset, batch_size=batch_size_classifier, shuffle=True)
    
    fed_trainloaders = []
    fed_valloaders = []
    
    gan_trainloaders = []
    
    for ds in datasets:
        
        all_indices = list()
        for i in range(len(ds)):
            all_indices.append(ds[i][1])
        
        for i in range(len(class_names)):
            mask = torch.tensor(all_indices) == i
            train_indices = mask.nonzero().reshape(-1)
            train_class = Subset(ds, train_indices)
            gan_trainloaders.append(DataLoader(train_class, batch_size=batch_size_gan, shuffle=True, drop_last=True))

        len_val = len(ds) // 10  
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        fed_trainloaders.append(DataLoader(ds_train, batch_size=batch_size_classifier, shuffle=True))
        fed_valloaders.append(DataLoader(ds_val, batch_size=batch_size_classifier, shuffle=True))
        
        
    testloader = DataLoader(testset, batch_size=batch_size_classifier)
    
    return gan_trainloaders, fed_trainloaders, fed_valloaders, testloader, benchmark_trainloader, benchmark_valloader, train_dataset