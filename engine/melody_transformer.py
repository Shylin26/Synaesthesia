import torch
import torch.nn as nn
class MelodyTransformer(nn.Module):
    def __init__(self,vocab_size=128,d_model=64,nhead=4,num_layers=3,num_emotions=4):
        super(MelodyTransformer,self).__init__()
        self.note_embedding=nn.Embedding(vocab_size,d_model)
        self.emotion_embedding=nn.Embedding(num_emotions,d_model)#embed into size 64 vetor
        self.pos_embedding=nn.Embedding(512,d_model)
        encoder_layer=nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        self.output_layer=nn.Linear(d_model,vocab_size)

    def forward(self,notes,emotion):
        batch_size,seq_len=notes.shape
        positions=torch.arange(seq_len,device=notes.device).unsqueeze(0)
        x=self.note_embedding(notes)
        x=x+self.pos_embedding(positions)
        x=x+self.emotion_embedding(emotion).unsqueeze(1)
        x=self.transformer(x)
        return self.output_layer(x)

def generate_melody(model,start_notes,emotion_id,length=20,temperature=1.0):
    model.eval()
    sequence=start_notes.clone()
    generated=[]
    emotion=torch.tensor([emotion_id])
    with torch.no_grad():
        for _ in range(length):
            output=model(sequence,emotion)
            # Prediction for last note in sequence
            logits=output[0,-1,:]/temperature
            probs=torch.softmax(logits,dim=0)
            next_note=torch.multinomial(probs,1).item()
            generated.append(next_note)
            # Append new note to sequence
            new_note=torch.tensor([[next_note]])
            sequence=torch.cat((sequence,new_note),dim=1)
    return generated

if __name__ == "__main__":
    model=MelodyTransformer()
    # Start with C major
    prompt=torch.tensor([[60,64,67,72]])
    for emotion_id , name in enumerate(["HAPPY","SAD","ANGRY","CALM"]):
        melody=generate_melody(model,prompt,emotion_id,length=8,temperature=0.8)
        print(f"{name}: {melody}")
