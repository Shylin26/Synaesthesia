import os
import sys
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from engine.emotion_transformer import EmotionTransformer

DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset.npz')
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'emotion_transformer.pt')
SCALER_SAVE_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')
EPOCHS=50
BATCH_SIZE=32
LR=0.001
def train():
    data=np.load(DATASET_PATH)
    X,y=data['X'],data['y']
    print(f"Loaded {X.shape[0]} samples.")

    scaler=StandardScaler()
    X=scaler.fit_transform(X)
    os.makedirs(os.path.dirname(SCALER_SAVE_PATH),exist_ok=True)
    with open(SCALER_SAVE_PATH,'wb') as f:
        pickle.dump(scaler,f)
    print("Scaler saved.")

    X_tensor=torch.tensor(X,dtype=torch.float32)
    y_tensor=torch.tensor(y,dtype=torch.long)
    dataset=TensorDataset(X_tensor,y_tensor)
    train_size=int(0.8*len(dataset))
    test_size=len(dataset)-train_size
    train_set,test_set=random_split(dataset,[train_size,test_size])
    train_loader=DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
    test_loader=DataLoader(test_set,batch_size=BATCH_SIZE)
    model=EmotionTransformer()
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    print("Start Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss=0
        for X_batch,y_batch in train_loader:
            predictions=model(X_batch)
            loss=criterion(predictions,y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        if(epoch+1)%10==0:
            model.eval()
            correct=0
            total=0
            with torch.no_grad():
                for X_batch,y_batch in test_loader:
                    preds=model(X_batch)
                    predicted=torch.argmax(preds,dim=1)
                    correct+=(predicted==y_batch).sum().item()
                    total+=y_batch.size(0)
            acc=100*correct/total
            avg_loss=total_loss/len(train_loader)
            print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | Test Acc: {acc:.1f}%")
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nTraining complete! Model saved to {MODEL_SAVE_PATH}")



if __name__=="__main__":
    train()