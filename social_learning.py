import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import copy
from timeit import default_timer as timer

#Exogenous parameteres
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


SMALL_EPOCHS=2000

BIG_EPOCHS=50
hidden_units=20

#Different learning rates for different models
lr_slow=3e-04
lr_not_so_slow=3e-03
lr_normal=3e-02
lr_fast=3e-01

#Data Generation

data=torch.arange(0,120,0.01).unsqueeze(dim=1)

#Hard data
#y=(data**0.5-134*torch.log(data+2)+8*torch.randn(size=data.size())+0.9*data**0.9)*10*torch.sin(data)+5*data-0.03*data**2+0.05*data**3-0.0007*data**4
#print(len(X)), print(y[:10])

#Very hard data
y=data**4-20*data**2+6-100*data**3+100*torch.sin(data)*data**2.2


#Creat training/test sets
train_split=int(0.8*len(data))
X_train,y_train=data[:train_split],y[:train_split]
X_test,y_test=data[train_split:],y[train_split:]


X_train,y_train=X_train.to(device),y_train.to(device)
X_test,y_test=X_test.to(device),y_test.to(device)


#General Model

class CultureHunter(nn.Module):
    
    def __init__(self, hidden_units:int):
        super().__init__()
        
        self.linear1=nn.Linear(in_features=1, out_features=hidden_units)
        self.linear2=nn.Linear(in_features=hidden_units, out_features=hidden_units)
        self.linear3=nn.Linear(in_features=hidden_units, out_features=1)
    
    def forward(self, x):
        x=self.linear1(x)
        x=F.relu(x)
        x=self.linear2(x)
        x=F.relu(x)
        x=self.linear3(x)
        
        return x

#Loss function
loss=nn.MSELoss()


#Instances of the same model with different learning rates

#Slow 
slow_hunter=CultureHunter(hidden_units=hidden_units).to(device)
optimizer_slow=torch.optim.Adam(params=slow_hunter.parameters(), lr=lr_slow)


#Not so slow
not_so_slow_hunter=CultureHunter(hidden_units=hidden_units).to(device)
optimizer_not_so_slow=torch.optim.Adam(params=not_so_slow_hunter.parameters(), lr=lr_not_so_slow)

#Normal 
normal_hunter=CultureHunter(hidden_units=hidden_units).to(device)
optimizer_normal=torch.optim.Adam(params=normal_hunter.parameters(), lr=lr_normal)

#Fast 
fast_hunter=CultureHunter(hidden_units=hidden_units).to(device)
optimizer_fast=torch.optim.Adam(params=fast_hunter.parameters(), lr=lr_fast)

#Lists for tracking results
loss_slow_all=[]
loss_not_so_slow_all=[]
loss_normal_all=[]
loss_fast_all=[]

test_loss_slow_all=[]
test_loss_not_so_slow_all=[]
test_loss_normal_all=[]
test_loss_fast_all=[]

big_epoch_n=[]

best_model_list=[]

start=timer()

#Training loop
for big_epoch in range(BIG_EPOCHS):
    
    #For each model we compute losses and deal usual changes aka small epoch training

    for epoch in range(SMALL_EPOCHS):
        
        
        #For slow model
        logits_slow=slow_hunter(X_train)
        loss_slow=loss(logits_slow, y_train)
        
        optimizer_slow.zero_grad(set_to_none=True)
        loss_slow.backward()
        optimizer_slow.step()
        
        if epoch==SMALL_EPOCHS:
            print (f"slow hunter losses: {loss_slow.log10()}")
    
    loss_slow_all.append(loss_slow.log10().item())       
            
    #For not so slow model

    for epoch in range(SMALL_EPOCHS):

        logits_not_so_slow=not_so_slow_hunter(X_train)
        loss_not_so_slow=loss(logits_not_so_slow, y_train)
            
        optimizer_not_so_slow.zero_grad(set_to_none=True)
        loss_not_so_slow.backward()
        optimizer_not_so_slow.step()
        
        if epoch==SMALL_EPOCHS:
            print (f"not so slow hunter losses: {loss_not_so_slow.log10()}")
    
    loss_not_so_slow_all.append(loss_not_so_slow.log10().item())    
        
    #For normal model

    for epoch in range(SMALL_EPOCHS):
        logits_normal=normal_hunter(X_train)
        loss_normal=loss(logits_normal, y_train)
            
        optimizer_normal.zero_grad(set_to_none=True)
        loss_normal.backward()
        optimizer_normal.step()
        
        if epoch==SMALL_EPOCHS:
            print (f"normal hunter losses: {loss_normal.log10()}")
    
    loss_normal_all.append(loss_normal.log10().item()) 
       
    #For fast model
    for epoch in range(SMALL_EPOCHS):
        logits_fast=fast_hunter(X_train)
        loss_fast=loss(logits_fast, y_train)
            
        optimizer_fast.zero_grad(set_to_none=True)
        loss_fast.backward()
        optimizer_fast.step()
        
        if epoch==SMALL_EPOCHS:
            print (f"fast hunter losses: {loss_fast.log10()}")
    
    loss_fast_all.append(loss_fast.log10().item()) 

    print(f"slow loss: {loss_slow.log10()} | not so slow loss: {loss_not_so_slow.log10()} | normal loss: {loss_normal.log10()} | fast loss: {loss_fast.log10()}")
            


    #Evaluation 
    with torch.no_grad():
        test_logits_slow=slow_hunter(X_test)
        test_loss_slow=loss(test_logits_slow,y_test)
        print(f'Slow hunter test loss: {test_loss_slow.log10()}')
        
        test_loss_slow_all.append(test_loss_slow.log10().item()) 
        
        test_logits_not_so_slow=not_so_slow_hunter(X_test)
        test_loss_not_so_slow=loss(test_logits_not_so_slow,y_test)
        print(f'Not so slow hunter test loss: {test_loss_not_so_slow.log10()}')
        
        test_loss_not_so_slow_all.append(test_loss_not_so_slow.log10().item()) 
        
        test_logits_normal=normal_hunter(X_test)
        test_loss_normal=loss(test_logits_normal,y_test)
        print(f'Normal hunter test loss: {test_loss_normal.log10()}')
        
        test_loss_normal_all.append(test_loss_normal.log10().item()) 
        
        test_logits_fast=fast_hunter(X_test)
        test_loss_fast=loss(test_logits_fast,y_test)
        print(f'Fast hunter test loss: {test_loss_fast.log10()}')
        
        test_loss_fast_all.append(test_loss_fast.log10().item()) 
    
    big_epoch_n.append(big_epoch)
    
    with torch.no_grad():
            
        #Create nested dictionary with computed losses
        dict_for_losses={}
        dict_for_losses['slow_hunter']= loss_slow
        dict_for_losses['not_so_slow_hunter']=loss_not_so_slow
        dict_for_losses['normal_hunter']=loss_normal
        dict_for_losses['fast_hunter']=loss_fast
                
        #Find minimum of losses and search for corresponding model name
        min_loss=min(dict_for_losses.values())
        min_dict = [key for key in dict_for_losses if dict_for_losses[key] == min_loss][0]
        print(min_dict)
        model_name=''.join(str(e) for e in min_dict)
        print(model_name)
        
        best_model_list.append(model_name)
        
        #Creating state.dicts for each model

        dict_for_models={}
        dict_for_models['slow_hunter']= copy.deepcopy(slow_hunter.state_dict())
        dict_for_models['not_so_slow_hunter']= copy.deepcopy(not_so_slow_hunter.state_dict())
        dict_for_models['normal_hunter']= copy.deepcopy(normal_hunter.state_dict())
        dict_for_models['fast_hunter']= copy.deepcopy(fast_hunter.state_dict())

        
        #Finding corresponding dict

        best_dict=dict_for_models.get(model_name)
        
        #Establish new point in parameters for all models

        slow_hunter.load_state_dict(best_dict)
        not_so_slow_hunter.load_state_dict(best_dict)
        normal_hunter.load_state_dict(best_dict)
        fast_hunter.load_state_dict(best_dict)
        
    

#Checking time (not necessary)
end=timer()
print(end-start)

#print(best_model_list)

#Writing file for best models list

with open('best_models_list.txt', 'w+') as f:
    
    # write elements of list
    for items in best_model_list:
        f.write('%s\n' %items)
    
    print("File written successfully")


f.close()

#Check length of data
print(len(test_loss_fast_all)), print(len(big_epoch_n))

with torch.no_grad():
    y_pred=slow_hunter(data.to(device))

#Plot model results
plt.plot(data.cpu().numpy(),y_pred.cpu().numpy(), label='Model')
plt.plot(data.cpu().numpy(),y.cpu().numpy(), label='Real Data')
plt.legend()
plt.show()

#Plot losses
plt.plot(big_epoch_n, loss_slow_all, label='Slow train loss')
plt.plot(big_epoch_n,test_loss_slow_all, label='Slow')
plt.plot(big_epoch_n,test_loss_not_so_slow_all, label='Not so slow')
plt.plot(big_epoch_n,test_loss_normal_all, label='Normal')
plt.plot(big_epoch_n,test_loss_fast_all, label='Fast')
plt.legend()
plt.show()