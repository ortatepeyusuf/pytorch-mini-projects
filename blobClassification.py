import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import torchmetrics

class blobModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=torch.nn.Sequential(torch.nn.Linear(in_features=2,out_features=10),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features=10, out_features=10),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features=10,out_features=4)
    )
    def forward(self,input):
        return self.layers(input)

    
class ModelTrainer:
    def __init__(self,epoch,model):
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.model=model.to(self.device)
        self.epochs=epoch
        x,y=make_blobs(n_samples=1000,n_features=2,centers=4)
        self.X_train,self.X_test,self.Y_train,self.Y_test=train_test_split(torch.from_numpy(x).to(dtype=torch.float32,device=self.device),
        torch.from_numpy(y).to(dtype=torch.float32,device=self.device),
        test_size=0.2,
        random_state=42)
        self.loss_func=torch.nn.CrossEntropyLoss()
        self.optimizer=torch.optim.Adam(params=self.model.parameters(),lr=0.01)
        self.train_loss=[]
        self.test_metrics=None
        self.acc=torchmetrics.Accuracy(task='multiclass',num_classes=4).to(device=self.device)
        self.prec=torchmetrics.Precision(task='multiclass',num_classes=4).to(device=self.device)
        self.f1=torchmetrics.F1Score(task='multiclass',num_classes=4).to(device=self.device)



    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            prediction=self.model(self.X_train).squeeze()
            loss=self.loss_func(prediction,self.Y_train.to(dtype=torch.long))
            self.train_loss.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def test(self):
        self.model.eval()
        with torch.inference_mode():
            pred=self.model(self.X_test)
            test_loss=self.loss_func(pred,self.Y_test.to(dtype=torch.long))
            test_pred=torch.softmax(pred,dim=1).argmax(dim=1)
            test_acc=self.acc(test_pred,self.Y_test.to(dtype=torch.long))
            test_prec=self.prec(pred,self.Y_test.to(dtype=torch.long))
            test_f1=self.f1(pred,self.Y_test.to(dtype=torch.long))
            self.test_metrics=f"Test Metrics\nLoss:{test_loss}\nAccuracy:{test_acc}\nPrecision:{test_prec}\nF1 Score:{test_f1[0]}"


    def visualize(self):
        plt.plot(range(self.epochs),self.train_loss,label='Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Value')
        print(self.test_metrics)
        plt.show()
        


    def main(self):
        self.train()
        self.test()
        self.visualize()

if __name__=='__main__':
    model=blobModel()
    trainer=ModelTrainer(100,model)
    trainer.main()




    

        

    