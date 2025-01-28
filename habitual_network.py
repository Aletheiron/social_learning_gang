import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from timeit import default_timer as timer

#Exogenous parameteres
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPOCHS=40000


hidden_units=20

lr=3e-04

#Data Generation

data=torch.arange(0,120,0.01).unsqueeze(dim=1)
y=(data**0.5-134*torch.log(data+2)+8*torch.randn(size=data.size())+0.9*data**0.9)*10*torch.sin(data)+5*data-0.03*data**2+0.05*data**3-0.0007*data**4

#Very Hard Data
#y=data**4-20*data**2+6-100*data**3+100*torch.sin(data)*data**2.2




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


#Instances of a model 

hunter=CultureHunter(hidden_units=hidden_units).to(device)
optimizer=torch.optim.Adam(params=hunter.parameters(), lr=lr)


#Train loop and tracking

loss_hab_hunter_all=[]
test_loss_hub_hunter_all=[]

epoch_n=[]

start=timer()

for epoch in range(EPOCHS):
    
    logits=hunter(X_train)
    loss_normal=loss(logits, y_train)
        
    optimizer.zero_grad(set_to_none=True)
    loss_normal.backward()
    optimizer.step()
    
    
    if epoch%100==0:
        print (f"normal hunter losses: {loss_normal.log10()}")
        loss_hab_hunter_all.append(loss_normal.log10().item())
        epoch_n.append(epoch)
        with torch.no_grad():
            test_logits=hunter(X_test)
            test_loss=loss(test_logits,y_test)
            print(f'normal hunter test loss: {test_loss.log10()}')
            test_loss_hub_hunter_all.append(test_loss.log10().item())

end=timer()
print(end-start)


print(len(test_loss_hub_hunter_all)), print(len(epoch_n))

with torch.no_grad():
    y_pred=hunter(data.to(device))

#Plot model results
plt.plot(data.cpu().numpy(),y_pred.cpu().numpy(), label='Model')
plt.plot(data.cpu().numpy(),y.cpu().numpy(), label='Real Data')
plt.legend()
plt.show()

#Plot losses
plt.plot(epoch_n, loss_hab_hunter_all, label='Train loss')
plt.plot(epoch_n,test_loss_hub_hunter_all, label='Test loss')
plt.legend()
plt.show()