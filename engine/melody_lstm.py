import torch
import torch.nn as nn

class MelodyLSTM(nn.Module):
    def __init__(self, vocab_size=128):
        # vocab_size=128 because there are 128 possible MIDI notes on a piano!
        super(MelodyLSTM, self).__init__()
        

        self.embedding = nn.Embedding(vocab_size, 64)
        
       
        self.lstm = nn.LSTM(input_size=64, hidden_size=256, num_layers=2, batch_first=True)
        
        
        self.dropout = nn.Dropout(0.3)
        
        
        self.fc = nn.Linear(256, vocab_size)
        
    def forward(self, x):
        """
        x is a sequence of notes. E.g. [C, E, G, G, E]
        """
        
        out = self.embedding(x)
        
        
        out, (hidden_state, cell_state) = self.lstm(out)
        
        
        out = self.dropout(out[:, -1, :])
        
        
        out = self.fc(out)
        return out
def generate_melody(model,start_sequence,length=20,temperature=1.0):
    """
    Given a starting sequence of notes, predict the next 'length' notes.
    Temperature > 1.0 makes it heavily random/creative.
    Temperature < 1.0 makes it very repetitive/safe.
    """
    model.eval()
    current_sequence=start_sequence.clone()
    generated_notes=[]
    with torch.no_grad():
        for _ in range(length):
            predictions=model(current_sequence)
            logits=predictions[0]/temperature
            probabilities=torch.softmax(logits,dim=0)
            next_note=torch.multinomial(probabilities,1).item()
            generated_notes.append(next_note)

            new_note_tensor=torch.tensor([[next_note]])
            current_sequence=torch.cat((current_sequence,new_note_tensor),dim=1)
    return generated_notes


if __name__ == "__main__":
    print("Initializing Generative AI...")
    model = MelodyLSTM()
    
    prompt = torch.tensor([[60, 62, 64, 65, 67]])
    
    print("\nStarting Prompt:", prompt[0].tolist())
    
    
    safe_melody = generate_melody(model, prompt, length=10, temperature=0.2)
    print("Safe AI Melody Generation:    ", safe_melody)
    
    wild_melody = generate_melody(model, prompt, length=10, temperature=2.0)
    print("Creative AI Melody Generation:", wild_melody)
