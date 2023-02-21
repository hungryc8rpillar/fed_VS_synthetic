import torch
import numpy as np
import matplotlib.pyplot as plt
from cellface import *
import matplotlib.pyplot as plt
from cellface.storage.container import *
import torch
import torchvision.utils as vutils
import torch.nn as nn
import opacus                                           
from opacus.validators import ModuleValidator
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from config import LATENT_DIM
from preprocessing import Dataset_Class
from utils import image_grid


def prepare_models(loaders,
                   class_names,
                   epochs,
                   batch_size,
                   latent_dim,
                   fixed_noise,
                   device,
                   num_partitions,
                   private,
                   epsilon,
                   delta,
                   cellface,
                   mnist,
                   ):  


    if num_partitions == 2:
        partition_names = ['Singapore','Munich']
    else:
        partition_names = []
        for i in range(num_partitions):
            partition_names.append('Partition ' + str(i+1))

    partition_counter = 0
    
    #generate dict for return values
    models = dict()
    
    
    
    
    for dataloader in loaders:
        reinitialization = True
        
        partition_name = partition_names[int((partition_counter-partition_counter%len(class_names))/len(class_names))]
        train_features, train_labels = next(iter(dataloader))
        key = str(class_names[int(train_labels[0])]) + 's from ' + partition_name
        print('****************************************************************************')
        print('Training now generator for ' + str(class_names[int(train_labels[0])]) + 's from ' + partition_name)
        
        while reinitialization:

            generator, discriminator, optimizer_G, optimizer_D, criterion, dataloader, privacy_engine = prepare_training(device,
                                                                                                                         dataloader,
                                                                                                                         epochs,
                                                                                                                         private,
                                                                                                                         epsilon,
                                                                                                                         delta,
                                                                                                                         cellface,
                                                                                                                         mnist,
                                                                                                                        )
    
            #img_list, G_losses, D_losses, generator, discriminator = train_gan(epochs, dataloader, latent_dim, batch_size, fixed_noise, generator, discriminator, optimizer_G, optimizer_D, criterion)
            models[key] = list(train_gan(reinitialization,
                                         epochs,
                                         dataloader,
                                         latent_dim,
                                         batch_size,
                                         fixed_noise,
                                         generator,
                                         discriminator,
                                         optimizer_G,
                                         optimizer_D,
                                         criterion,
                                         device,
                                         private,
                                         privacy_engine,
                                         delta))
            print('\n')
            if models[key] != list('initialize_signal'):
                reinitialization = False
                generation_noise = torch.randn(batch_size, latent_dim, 1,1).to(device)
                generated = generator(generation_noise).detach().cpu() 
                models[key].append(generated) #fake images
                models[key].append(train_features[0:0+batch_size]) #real images
                models[key].append(str(class_names[int(train_labels[0])])) #label
        partition_counter += 1
    return models
        

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            
            nn.ConvTranspose2d(LATENT_DIM, 96 * 8, 6, 1, 0, bias=False),
            nn.BatchNorm2d(96 * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(96 * 8, 96 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96 * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(96 * 4, 96 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96 * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(96 * 2, 96 , 4, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(96, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            
        )        

    def forward(self, input):
        return self.main(input)
    
class MNIST_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            
            nn.ConvTranspose2d(LATENT_DIM, 28 * 8, 4, 2, 0, bias=False),
            nn.BatchNorm2d(28 * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(28 * 8, 28 * 4, 4, 2, 0, bias=False),
            nn.BatchNorm2d(28 * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(28 * 4, 28 * 2,4, 2, 0, bias=False),
            nn.BatchNorm2d(28 * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(28 * 2, 28 * 1, 4, 1, 0, bias=False),
            nn.BatchNorm2d(28 * 1),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(28 * 1,1, 4, 1, 0, bias=False),
            nn.Tanh()
            
            
        )        

    def forward(self, input):
        return self.main(input)
    

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            
            nn.Conv2d( 1, 96, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d( 96, 96*2, 4, 2, 0, bias=False),
            nn.GroupNorm(32,96*2),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d( 96*2, 96*4, 4, 2, 0, bias=False),
            nn.GroupNorm(32,96*4),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d( 96*4, 96*8, 4, 2, 0, bias=False),
            nn.GroupNorm(32,96*8),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d( 96*8, 1, 4, 2, 0, bias=False),
            nn.Sigmoid()
    
        )
    

    def forward(self, input):
        return self.main(input)
    
class MNIST_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            
            nn.Conv2d( 1, 28, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d( 28, 28*2, 2, 2, 0, bias=False),
            nn.GroupNorm(7,28*2),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d( 28*2, 28*4, 2, 2, 0, bias=False),
            nn.GroupNorm(7,28*4),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d( 28*4, 1, 2, 2, 0, bias=False),
            nn.Sigmoid()
    
        )

    def forward(self, input):
        return self.main(input)

def prepare_training(device,
                     dataloader,
                     epochs,
                     private,
                     epsilon,
                     delta,
                     cellface,
                     mnist,
                    ):

    if cellface:
        generator = Generator()
    if mnist:
        generator = MNIST_Generator()
    generator.to(device)   
    #generator.eval()
    generator.train()

    if cellface:
        discriminator = Discriminator()
    if mnist:
        discriminator = MNIST_Discriminator()
    discriminator.to(device) 
    #discriminator.eval()
    discriminator.train()
    
    
    discriminator = ModuleValidator.fix(discriminator)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0008, betas=(0.5, 0.999)) #celllr=0.0002 mnist*10
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0008, betas=(0.5, 0.999))

    criterion = nn.BCELoss()
    
    privacy_engine = None
    if private: #make only discriminator private because generator never sees real sample
        privacy_engine = opacus.PrivacyEngine()

        discriminator, optimizer_D, dataloader = privacy_engine.make_private_with_epsilon(
            module=discriminator,
            optimizer=optimizer_D,
            data_loader=dataloader,
            target_epsilon=epsilon,
            target_delta=delta,
            epochs = epochs,
            max_grad_norm=0.2,
        )
        
    discriminator.to(device) 

    return generator, discriminator, optimizer_G, optimizer_D, criterion, dataloader, privacy_engine


def train_gan(reinitialization,
              epochs,
              dataloader,
              latent_dim,
              batch_size,
              fixed_noise,
              generator,
              discriminator,
              optimizer_G,
              optimizer_D,
              criterion,
              device,
              private,
              privacy_engine,
              delta):
    
    
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    real_label = 1.
    fake_label = 0.
    initialization_bound = epochs - 1


    for epoch in range(epochs):

        if epoch > 0 and epoch < initialization_bound and (G_losses[-1] > 70 or D_losses[-1] > 70):
            print('\n Generator crashed... reinitialization triggered')
            break

        if epoch == initialization_bound:
            reinitialization = False

        for i, data in enumerate(dataloader, 0):
            optimizer_D.zero_grad()
            real_ondevice = data[0].to(device)

            if real_ondevice.size(0) == 0:
                real_ondevice = lastround_real #replace empty batch with last round
            lastround_real = real_ondevice #save batch to use again if next is empty
    
            dynamic_batch_size = real_ondevice.size(0)
            dynamic_real_label = torch.full((dynamic_batch_size,), real_label, dtype=torch.float, device=device)
            output = discriminator(real_ondevice).view(-1)

            errD_real = criterion(output, dynamic_real_label)

            if not private:
                errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(dynamic_batch_size, latent_dim, 1, 1, device=device)

            fake = generator(noise)
            dynamic_fake_label = torch.full((dynamic_batch_size,), fake_label, dtype=torch.float, device=device)

            output = discriminator(fake.detach()).view(-1)
            errD_fake = criterion(output, dynamic_fake_label)
            
            if private:
                errD = errD_real + errD_fake
                errD.backward()
                optimizer_D.step()
                optimizer_D.zero_grad(set_to_none=True)
                D_G_z1 = output.mean().item()
            
            else:
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                
                optimizer_D.step()
                

            
            optimizer_G.zero_grad()
            output = discriminator(fake).view(-1)
            errG = criterion(output, dynamic_real_label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizer_G.step()

            G_losses.append(errG.item())
            D_losses.append(errD.item())
            

        if epoch%5==0 and not private:   
            print('[%d/%d]\tD Loss: %.4f\tG Loss: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch+1, epochs,
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2)) 
        if epoch%5==0 and private:
            print('[%d/%d]\tD Loss: %.4f\tG Loss: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch+1, epochs,
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            epsilon = privacy_engine.accountant.get_epsilon(delta=delta)
            print('Privacy budget: ε = %.2f, δ = %.6f' % (epsilon, delta))
            
        with torch.no_grad():
            fake = generator(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    if reinitialization == True:
        return 'initialize_signal'
        
    return img_list, G_losses, D_losses, generator, discriminator

    
def show_gan_results(models): 
    for key in models:  #content of each dict entry: img_list, G_losses, D_losses, generator, discriminator, fake data, real data, label
       #key = 'PLTs from Munich'
        this_G_loss = models[key][1]
        this_D_loss = models[key][2]
        this_gen = models[key][3]
        this_dis = models[key][4]
        fake = models[key][5]
        real = models[key][6]
        label = models[key][7]

        real_images = image_grid(np.transpose(np.array(real),(0,2,3,1)))
        print(real_images.shape)
        fake_images = image_grid(np.transpose(np.array(fake),(0,2,3,1)))
        fig = plt.figure(figsize=(20., 20.))
        plt.subplot(1,2,1)
        plt.axis("off")
        plt.title('Real ' + str(key))
        plt.imshow(real_images)
        plt.subplot(1,2,2)
        plt.axis("off")
        plt.title('Generated '+ str(key))
        plt.imshow(fake_images)

        plt.figure(figsize=(16.,3.))
        plt.title("G and D loss for " + str(key))
        plt.plot(this_G_loss,label="G")
        plt.plot(this_D_loss,label="D")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        
def generate_classification_data(models, device, latent_dim, class_names, batch_size, train_dataset):
    
    trainingsset = list()
    validationset = list()
    labelcounter = 0
    
    for key in models:  

        this_G_loss = models[key][1]
        this_D_loss = models[key][2]
        this_gen = models[key][3]
        this_dis = models[key][4]
        fake = models[key][5]
        real = models[key][6]
        label = models[key][7]
        
        
        len_synthetic_partition =len(train_dataset) // len(models)
        len_val = len_synthetic_partition // 10
        len_train = len_synthetic_partition - len_val

        train_noise = torch.randn(len_train, latent_dim, 1,1).to(device)
        val_noise = torch.randn(len_val, latent_dim, 1,1).to(device)
        this_gen.to(device)

        train_synthesis = this_gen(train_noise)
        train_synthesis = np.squeeze(train_synthesis)

        val_synthesis = this_gen(val_noise)
        val_synthesis = np.squeeze(val_synthesis)


        train_features = list()
        for img in train_synthesis:
            train_features.append(img.cpu().detach().numpy())

        val_features = list()
        for img in val_synthesis:
            val_features.append(img.cpu().detach().numpy())

        trainingsset.append(Dataset_Class(train_features, labelcounter%len(class_names)))
        validationset.append(Dataset_Class(val_features, labelcounter%len(class_names)))

        readytotrain = torch.utils.data.ConcatDataset(trainingsset)
        readytoval = torch.utils.data.ConcatDataset(validationset)

        labelcounter += 1

    classifier_trainloader = DataLoader(readytotrain, batch_size=batch_size, shuffle=True)
    classifier_valloader = DataLoader(readytoval, batch_size=batch_size, shuffle=True)
    return classifier_trainloader, classifier_valloader