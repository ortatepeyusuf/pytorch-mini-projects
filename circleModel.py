import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

class circleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.lin1=torch.nn.Linear(in_features=2,out_features=10,device=self.device)
        self.lin2=torch.nn.Linear(in_features=10,out_features=10,device=self.device)
        self.lin3=torch.nn.Linear(in_features=10,out_features=1,device=self.device)
        self.relu=torch.nn.ReLU()
        

    def forward(self,input):
        return self.lin3(self.relu(self.lin2(self.relu(self.lin1(input)))))

class ModelTrainer:
    def __init__(self,model,epoch):
        self.model=model
        self.loss_func=torch.nn.BCEWithLogitsLoss()
        self.optimizer=torch.optim.Adam(params=self.model.parameters(),lr=0.01)
        self.epochs=epoch
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        x,y= make_circles(1000,
                    noise=0.03, 
                    random_state=42)
        
        self.X_train,self.X_test,self.Y_train,self.Y_test=train_test_split(torch.from_numpy(x).to(dtype=torch.float32,device=self.device),
        torch.from_numpy(y).to(dtype=torch.float32,device=self.device),
        test_size=0.2,
        random_state=42)
        self.train_loss=[]
        self.test_values=""


    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            prediction=self.model(self.X_train).squeeze()
            loss=self.loss_func(prediction,self.Y_train)
            self.train_loss.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # if epoch%100==0:
            #     print(f'Epoch:{epoch} Training Loss:{loss.item()}')

    def test_and_acc(self):
        self.model.eval()
        with torch.inference_mode():
            prediction=self.model(self.X_test).squeeze()
            test_loss=self.loss_func(prediction,self.Y_test)
        probabilites=torch.sigmoid(prediction)
        labels=torch.round(probabilites)
        correctness=torch.eq(labels,self.Y_test).sum().item()
        acc=correctness/len(self.Y_test)*100
        self.test_values=f"Test Values\nTest Loss:{test_loss}\nTest Accuracy:{acc}"

    
    def visualize(self):
        plt.plot(range(self.epochs),self.train_loss,label='Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Value')
        plt.show()
        print(self.test_values)

    def main(self):
        self.train()
        self.test_and_acc()
        self.visualize()    
        

    
if __name__ == '__main__':
    model=circleModel()
    trainer=ModelTrainer(model,100)
    trainer.main()

