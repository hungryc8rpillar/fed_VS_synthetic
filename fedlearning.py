import torch
import numpy as np
from opacus import PrivacyEngine
from cellface import *
from cellface.storage.container import *
from collections import OrderedDict

import flwr as fl
import numpy as np
import matplotlib.pyplot as plt
from flwr.common import Metrics

from config import FED_DEVICE,DELTA, FED_EPOCHS, FED_ROUNDS, FED_MODEL_PATH, EPSILON
from classifier import test


def evaluate_config(server_round: int):
    return {'server_round': server_round,
           'rounds': FED_ROUNDS,
           'model_path': FED_MODEL_PATH}

def fit_config(server_round: int):
    config = {
        'server_round': server_round, 
        'private': False,
        'epochs' : FED_EPOCHS,
        'rounds' : FED_ROUNDS,
        'aimed_epsilon': 50,
        'delta': DELTA, # might have to make it smaller for federated, e.g. delta/partitions
        'server_round': server_round,
    }
    return config



def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    
def train_fed(net, trainloader, privacy_engine, local_epochs, global_rounds, private, aimed_epsilon, delta):
    
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    if private:
        
        net, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
                module=net,
                optimizer=optimizer,
                data_loader=trainloader,
                target_epsilon=aimed_epsilon,
                target_delta=delta,
                epochs = local_epochs * global_rounds,
                max_grad_norm=0.2,
            )

    epsilon = 0
    
    for epoch in range(local_epochs):
        for images, labels in trainloader:
            images, labels = images.to(FED_DEVICE), labels.to(FED_DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
    if private:        
        epsilon = privacy_engine.accountant.get_epsilon(delta=DELTA)
    
    return epsilon

def get_privacy_engine(cid, PE):
    if cid not in PE.keys():
        PE[cid] = PrivacyEngine()
    return PE[cid] 

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader, privacy_engine):
        super().__init__()
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.privacy_engine = privacy_engine

    def get_parameters(self, config):
        return get_parameters(self.net)
    
    def fit(self, parameters, config):
        
        private = config['private']
        local_epochs = config['epochs']
        global_rounds = config['rounds']
        aimed_epsilon = config['aimed_epsilon']
        delta = config['delta']
        
        set_parameters(self.net, parameters)
        
        epsilon = train_fed(self.net,
                            self.trainloader,
                            self.privacy_engine,
                            local_epochs,
                            global_rounds,
                            private = private,
                            aimed_epsilon = aimed_epsilon,
                            delta = delta,
                           )
        #TODO: tracking of privacy budget
        #if private:
         #   print(f"[CLIENT {self.cid}] epsilon = {epsilon:.2f}")
        return get_parameters(self.net), len(self.trainloader), {"epsilon":epsilon}

    def evaluate(self, parameters, config):
        
        serverround = config['server_round']
        global_rounds = config['rounds']
        model_path = config['model_path']
        
        set_parameters(self.net, parameters)
        train_loss, train_accuracy = test(self.net, self.trainloader, FED_DEVICE)
        val_loss, val_accuracy = test(self.net, self.valloader, FED_DEVICE)
        print(f'Round: {serverround}, [CLIENT {self.cid}]')
        print(f'Train loss: {train_loss}, Train accuracy: {train_accuracy}')
        print(f'Validation loss: {val_loss}, Validation accuracy: {val_accuracy}\n')
        
        if serverround == global_rounds:
            print('Training finished, saving model \n')
            torch.save(self.net.state_dict(), model_path) 
        
        return (float(val_loss),
                len(self.trainloader),
                {
                'train_accuracy': float(train_accuracy),
                'val_accuracy' : float(val_accuracy),
                'train_loss' : float(train_loss),
                'val_loss' : float(val_loss),
                },
               )



def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    train_accuracies = [num_examples * m['train_accuracy'] for num_examples, m in metrics]
    val_accuracies = [num_examples * m['val_accuracy'] for num_examples, m in metrics]
    train_losses = [num_examples * m['train_loss'] for num_examples, m in metrics]
    val_losses = [num_examples * m['val_loss'] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {
            'train_loss': sum(train_losses) / sum(examples),
            'train_accuracy': sum(train_accuracies) / sum(examples),
            'val_loss': sum(val_losses) / sum(examples),
            'val_accuracy': sum(val_accuracies) / sum(examples),
           }
        
public_fedavg_strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  
        fraction_evaluate=1.0,  
        min_fit_clients=2,  
        min_evaluate_clients=2,  
        min_available_clients=2,  
        evaluate_metrics_aggregation_fn=weighted_average,
        on_evaluate_config_fn=evaluate_config,
        on_fit_config_fn=fit_config,
       # proximal_mu = 1000
)


public_fedprox_strategy = fl.server.strategy.FedProx(
        fraction_fit=1.0,  
        fraction_evaluate=1.0,  
        min_fit_clients=2,  
        min_evaluate_clients=2,  
        min_available_clients=2,  
        evaluate_metrics_aggregation_fn=weighted_average,
        on_evaluate_config_fn=evaluate_config,
        on_fit_config_fn=fit_config,
        proximal_mu = 1000
)

private_strategies = dict()

for entry in EPSILON:
    def fit_config(server_round: int):
        config = {
            'server_round': server_round, 
            'private': True,
            'epochs' : FED_EPOCHS,
            'rounds' : FED_ROUNDS,
            'aimed_epsilon': entry,
            'delta': DELTA, # might have to make it smaller for federated, e.g. delta/partitions
            'server_round': server_round,
        }
        return config
    
    fedavg_strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,  
            fraction_evaluate=1.0,  
            min_fit_clients=2,  
            min_evaluate_clients=2,  
            min_available_clients=2,  
            evaluate_metrics_aggregation_fn=weighted_average,
            on_evaluate_config_fn=evaluate_config,
            on_fit_config_fn=fit_config,
        # proximal_mu = 1000
        )


    fedprox_strategy = fl.server.strategy.FedProx(
            fraction_fit=1.0,  
            fraction_evaluate=1.0,  
            min_fit_clients=2,  
            min_evaluate_clients=2,  
            min_available_clients=2,  
            evaluate_metrics_aggregation_fn=weighted_average,
            on_evaluate_config_fn=evaluate_config,
            on_fit_config_fn=fit_config,
            proximal_mu = 1
        )
        
    private_strategies[entry] = [fedavg_strategy,fedprox_strategy]
