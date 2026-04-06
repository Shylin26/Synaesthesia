import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from engine.melody_transformer import MelodyTransformer

MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'melody_transformer.pt')
EPOCHS = 60
BATCH_SIZE = 32
LR = 0.001
SEQ_LEN = 16

SCALES = {
    0: [60, 62, 64, 65, 67, 69, 71, 72],  # HAPPY  → C major
    1: [57, 59, 60, 62, 64, 65, 67, 69],  # SAD    → A minor
    2: [60, 61, 62, 63, 64, 65, 66, 67],  # ANGRY  → chromatic
    3: [60, 62, 64, 67, 69, 72, 74, 76],  # CALM   → C pentatonic
}

def generate_sequences(emotion_id, num_sequences=500):
    scale = SCALES[emotion_id]
    sequences = []
    for _ in range(num_sequences):
        seq = []
        note = np.random.choice(scale)
        for _ in range(SEQ_LEN + 1):
            seq.append(note)
            step = np.random.choice([-2, -1, 0, 1, 2],
                                     p=[0.1, 0.25, 0.1, 0.25, 0.3])
            idx = scale.index(note) if note in scale else 0
            next_idx = np.clip(idx + step, 0, len(scale) - 1)
            note = scale[next_idx]
        sequences.append(seq)  
    return sequences           

def build_melody_dataset():
    all_inputs, all_targets, all_emotions = [], [], []
    for emotion_id in range(4):
        sequences = generate_sequences(emotion_id)
        for seq in sequences:
            all_inputs.append(seq[:-1])  
            all_targets.append(seq[1:])
            all_emotions.append(emotion_id)
    X = torch.tensor(all_inputs, dtype=torch.long)
    y = torch.tensor(all_targets, dtype=torch.long)
    emotions = torch.tensor(all_emotions, dtype=torch.long)
    return X, y, emotions

def train():
    print("Building melody dataset...")
    X, y, emotions = build_melody_dataset()
    print(f"Dataset: {X.shape[0]} sequences of {X.shape[1]} notes each.")

    dataset = TensorDataset(X, y, emotions)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MelodyTransformer()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # fixed: model_parameters()

    print("Training melody model...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0  # fixed: was train_loss
        for X_batch, y_batch, e_batch in loader:
            output = model(X_batch, e_batch)
            loss = criterion(
                output.view(-1, 128),
                y_batch.view(-1)   # fixed: was .vie(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nMelody model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":   
    train()
