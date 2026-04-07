import torch
import torch.nn as nn
from pathlib import Path
class MelodyTransformer(nn.Module):
    def __init__(self, vocab_size=128, d_model=128, nhead=8, num_layers=4, num_emotions=4):
        super(MelodyTransformer, self).__init__()
        self.note_embedding = nn.Embedding(vocab_size, d_model)
        self.emotion_embedding = nn.Embedding(num_emotions, d_model)
        self.pos_embedding = nn.Embedding(512, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self,notes,emotion):
        batch_size,seq_len=notes.shape
        positions=torch.arange(seq_len,device=notes.device).unsqueeze(0)
        x=self.note_embedding(notes)
        x=x+self.pos_embedding(positions)
        x=x+self.emotion_embedding(emotion).unsqueeze(1)
        x=self.transformer(x)
        return self.output_layer(x)

def generate_melody(model, start_notes, emotion_id, length=20, temperature=1.0, allowed_notes=None):
    model.eval()
    sequence = start_notes.clone()
    generated = []
    emotion = torch.tensor([emotion_id])
    recent = []

    EMOTION_SCALES = {
        0: [60,62,64,65,67,69,71,72,74,76,77,79],
        1: [57,59,60,62,64,65,67,69,71,72,74,76],
        2: [60,61,63,65,66,68,70,72,73,75,77,78],
        3: [60,62,64,67,69,72,74,76,79,81],
    }
    scale = allowed_notes if allowed_notes else EMOTION_SCALES.get(emotion_id, list(range(48,85)))
    scale = [n for n in scale if 48 <= n <= 84]

    mask = torch.full((128,), float('-inf'))
    for note in scale:
        mask[note] = 0.0

    with torch.no_grad():
        for _ in range(length):
            output = model(sequence, emotion)
            logits = output[0, -1, :] / temperature
            logits = logits + mask
            for prev in set(recent[-3:]):
                logits[prev] -= 3.0
            probs = torch.softmax(logits, dim=0)
            next_note = torch.multinomial(probs, 1).item()
            generated.append(next_note)
            recent.append(next_note)
            sequence = torch.cat((sequence, torch.tensor([[next_note]])), dim=1)

    return generated

if __name__ == "__main__":
    model=MelodyTransformer()
    # Start with C major
    prompt=torch.tensor([[60,64,67,72]])
    for emotion_id , name in enumerate(["HAPPY","SAD","ANGRY","CALM"]):
        melody=generate_melody(model,prompt,emotion_id,length=8,temperature=0.8)
        print(f"{name}: {melody}")
