import torch
import torch.nn as nn

class EmotionConditionedLSTM(nn.Module):
    def __init__(self, vocab_size=128, emotion_dim=4):
        """
        emotion_dim=4 to ingest our 4 core extracted parameters:
        [syncopation, dissonance, arpeggiation, groove]
        """
        super(EmotionConditionedLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, 64)
        # We increase LSTM input shape to accept the concatenated emotion vector
        self.lstm = nn.LSTM(input_size=64 + emotion_dim, hidden_size=256, num_layers=3, batch_first=True, dropout=0.2)
        
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, vocab_size)
        )
        
    def forward(self, x, emotion_vector):
        """
        x: sequence of notes [batch, seq_len]
        emotion_vector: [batch, emotion_dim]
        """
        # Embed notes into dense vectors
        notes_embedded = self.embedding(x) # [batch, seq_len, 64]
        
        # We must broadcast the emotion vector across the entire time sequence
        # [batch, emotion_dim] -> [batch, 1, emotion_dim] -> [batch, seq_len, emotion_dim]
        emotion_expanded = emotion_vector.unsqueeze(1).repeat(1, notes_embedded.size(1), 1)
        
        # Mathematically fuse the music notes with the emotional landscape state
        lstm_input = torch.cat([notes_embedded, emotion_expanded], dim=-1)
        
        out, (h, c) = self.lstm(lstm_input)
        
        # Grab the last sequencer step to predict the next note
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def generate_conditional_melody(model, start_sequence, emotion_vector, length=20, temperature=1.0):
    model.eval()
    current_sequence = start_sequence.clone()
    generated_notes = []
    
    with torch.no_grad():
        for _ in range(length):
            # Predict the next note condition on the emotion vector
            logits = model(current_sequence, emotion_vector)
            
            # Apply creativity temperature control
            logits = logits[0] / temperature
            probabilities = torch.softmax(logits, dim=0)
            
            # Sample the distribution
            next_note = torch.multinomial(probabilities, 1).item()
            generated_notes.append(next_note)

            # Append the new note and slide the window for the next autoregressive prediction
            new_note_tensor = torch.tensor([[next_note]])
            current_sequence = torch.cat((current_sequence, new_note_tensor), dim=1)
            
    return generated_notes

if __name__ == "__main__":
    print("Testing Masterpiece Conditional Architecture Initialization...")
    model = EmotionConditionedLSTM()
    
    prompt = torch.tensor([[60, 62, 64, 65, 67]])
    # Mocking a "Neon Nostalgia" matrix: high syncopation, low dissonance, high arp, mid groove
    neon_nostalgia_params = torch.tensor([[0.8, 0.1, 0.9, 0.4]])
    
    print("\nPrompt Notes:", prompt[0].tolist())
    print("Emotion Condition (S,D,A,G):", neon_nostalgia_params[0].tolist())
    
    generated = generate_conditional_melody(model, prompt, neon_nostalgia_params, length=12, temperature=1.2)
    print("\nDreamt Conditional Melody:", generated)
