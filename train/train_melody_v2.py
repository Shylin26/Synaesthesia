import os, sys, torch, torch.nn as nn, numpy as np
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from engine.melody_transformer import MelodyTransformer

DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'music_dataset.npz')
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'melody_transformer.pt')
EPOCHS = 120
BATCH_SIZE = 64
LR = 0.0005

def train():
    data = np.load(DATASET_PATH)
    X = torch.tensor(data['X'], dtype=torch.long)
    y = torch.tensor(data['y'], dtype=torch.long)
    emotions = torch.tensor(data['emotions'], dtype=torch.long)
    print(f"Dataset: {X.shape[0]} sequences of {X.shape[1]} notes.")
    loader=DataLoader(TensorDataset(X,y,emotions),batch_size=BATCH_SIZE,shuffle=True)
    model=MelodyTransformer()
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(EPOCHS):
        model.train()
        total_loss=0
        for X_b,y_b,e_b in loader:
            out=model(X_b,e_b)
            loss=criterion(out.view(-1,128),y_b.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        scheduler.step()
        if(epoch+1)%20==0:
            print(f"Epoch {epoch+1:03d}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH),exist_ok=True)
    torch.save(model.state_dict(),MODEL_SAVE_PATH)
    print(f"Saved -> {MODEL_SAVE_PATH}")

if __name__=="__main__":
    train()



