import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset,random_split
from sklearn.preprocessing import StandardScaler
import pickle
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from engine.emotion_regressor import EmotionRegressor
DATASET_PATH=os.path.join(os.path.dirname(__file__),'..','data','deam_dataset.npz')
MODEL_SAVE_PATH=os.path.join(os.path.dirname(__file__),'..','models','emotion_regressor.pt')
SCALER_SAVE_PATH=os.path.join(os.path.dirname(__file__),'..','models','deam_scaler.pkl')
EPOCHS=80
BATCH_SIZE=32
LR=0.001
def train():
    data=np.load(DATASET_PATH)
    X,valence,arousal=data['X'],data['valence'],data['arousal']
    print(f"Loaded{X.shape[0]}samples.")
    scaler=StandardScaler()
    X=scaler.fit_transform(X)
    os.makedirs(os.path.dirname(SCALER_SAVE_PATH),exist_ok=True)
    with open(SCALER_SAVE_PATH,'wb') as f:
        pickle.dump(scaler,f)
    
    X_t=torch.tensor(X,dtype=torch.float32)
    y_t=torch.tensor(np.stack([valence,arousal],axis=1),dtype=torch.float32)
    dataset=TensorDataset(X_t,y_t)
    train_size=int(0.8*len(dataset))
    test_size=len(dataset)-train_size
    train_set,test_set=random_split(dataset,[train_size,test_size])
    train_loader=DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
    test_loader=DataLoader(test_set,batch_size=BATCH_SIZE)

    model = EmotionRegressor(feature_dim=176)
    criterion=nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)

    print("Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss=0
        for X_batch,y_batch in train_loader:
            pred=model(X_batch)
            loss=criterion(pred,y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        
        if(epoch+1)%10==0:
            model.eval()
            mae_v,mae_a,n=0,0,0
            with torch.no_grad():
                for X_batch,y_batch in test_loader:
                    pred=model(X_batch)
                    mae_v+=torch.abs(pred[:,0] - y_batch[:,0]).sum().item()
                    mae_a += torch.abs(pred[:,1] - y_batch[:,1]).sum().item()
                    n+=y_batch.size(0)
            
            avg_loss=total_loss/len(train_loader)
            print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | MAE V: {mae_v/n:.3f} A: {mae_a/n:.3f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train() 





