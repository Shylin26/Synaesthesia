import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pretty_midi
import glob

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from engine.melody_lstm import EmotionConditionedLSTM

class MidiEmotionDataset(Dataset):
    def __init__(self, midi_folder, seq_len=20):
        self.samples = []
        self.seq_len = seq_len
        
        print(f"Scanning '{midi_folder}' for MIDI files to train the Network...")
        
        if not os.path.exists(midi_folder):
            print(f"WARNING: Folder '{midi_folder}' not found. Generating a mathematically perfect dummy dataset to verify architecture...")
            
            for _ in range(100):
                self.samples.append({
                    "notes": torch.randint(60, 75, (self.seq_len + 1,)), # Random safe notes
                    "emotion": torch.rand(4, dtype=torch.float32) # [S,D,A,G] vector shape
                })
            return
            
        for midi_file in glob.glob(os.path.join(midi_folder,"**/*.mid"),recursive=True):
            try:
                pm=pretty_midi.PrettyMIDI(midi_file)
                instrument=pm.instruments[0]
                sorted_notes=sorted(instrument.notes,key=lambda n: n.start)
                pitches=[note.pitch for note in sorted_notes]
                for i in range(len(pitches)-self.seq_len):
                    chunk=pitches[i:i+self.seq_len+1]
                    emotion_vector = torch.tensor([0.8, 0.2, 0.9, 0.4], dtype=torch.float32)
                    self.samples.append({
                        "notes":torch.tensor(chunk,dtype=torch.long),
                        "emotion":emotion_vector

                    })
            except Exception as e:
                print(f"Skipping corrupted MIDI {midi_file}: {e}")
        print(f"Successfully chunked {len(self.samples)} sequences for training!")






    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        notes = chunk["notes"]
        
        # x is the sequence up to the last note
        # y is the exact single next note we want the network to predict
        x = notes[:-1]
        y = notes[-1]
        emotion = chunk["emotion"]
        
        return x, y, emotion

def train_lstm_composer(data_folder="my_midi_bank", epochs=50):
    dataset = MidiEmotionDataset(data_folder)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Initialize our masterpiece architecture
    model = EmotionConditionedLSTM(vocab_size=128, emotion_dim=4)
    model.train()
    
    # CrossEntropy because we are classifying which out of 128 keys to hit next
    criterion = nn.CrossEntropyLoss()
    
    # AdamW stabilizes the highly-variant embeddings inside the LSTM
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print("\n--- SYNAPSE ENGAGED: TRAINING LSTM COMPOSER ---")
    for epoch in range(epochs):
        total_loss = 0
        for x, y, emotion in loader:
            optimizer.zero_grad()
            
            # Feed the notes AND the emotion constraints into the beast
            logits = model(x, emotion) 
            
            loss = criterion(logits, y)
            loss.backward()
            
            # Critical step to prevent exploding gradients in recurrent networks
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Creativity Loss: {total_loss/len(loader):.4f}")
            
    # Export it where the API can automatically discover it
    os.makedirs("../models", exist_ok=True)
    torch.save(model.state_dict(), "../models/melody_lstm.pt")
    print("\n[SUCCESS] Saved Neural Composer Checkpoint to 'models/melody_lstm.pt'!")

if __name__ == "__main__":
    train_lstm_composer()
