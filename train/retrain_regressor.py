import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from engine.emotion_regressor import EmotionRegressor
def train_masterpiece_model(X_train_np,Y_train_np):
    model=EmotionRegressor(feature_dim=176,d_model=128,nhead=8,num_layers=4)
    model.train()
    criterion=nn.MSELoss()
    optimizer=optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50)
    dataset=TensorDataset(torch.tensor(X_train_np,dtype=torch.float32),
                          torch.tensor(Y_train_np,dtype=torch.float32))
    loader=DataLoader(dataset,batch_size=32,shuffle=True)
    print("Training this Architecture...")
    for epoch in range(50):
        total_loss=0
        for batch_x,batch_y in loader:
            optimizer.zero_grad()
            predictions=model(batch_x)
            loss=criterion(predictions,batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            optimizer.step()
            total_loss+=loss.item()
        scheduler.step()
        if (epoch+1)%5==0:
            print(f"Epoch {epoch+1}/50, Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(),"../models/emotion_regressor.pt")
    print("Saved as models/emotion_regressor.pt")

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    
    print("Generating dummy 176-dim data to test the pipeline...")
    
    X_dummy = np.random.randn(100, 176) 
    Y_dummy = np.random.uniform(-1, 1, size=(100, 2)) 
    

    train_masterpiece_model(X_dummy, Y_dummy)
