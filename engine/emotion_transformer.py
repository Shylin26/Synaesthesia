import torch
import torch.nn as nn
class EmotionTransformer(nn.Module):
    def __init__(self,feature_dim=128,d_model=64,nhead=4,num_layers=2,num_emotions=4):
        super(EmotionTransformer,self).__init__()
        self.input_proj=nn.Linear(feature_dim,d_model)
        encoder_layer=nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        self.classifier=nn.Sequential(
            nn.Linear(d_model,32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32,num_emotions)
        )
    
    def forward(self,x):
        x=self.input_proj(x)
        x=x.unsqueeze(1)
        x=self.transformer(x)
        x=x.squeeze(1)
        return self.classifier(x)

if __name__=="__main__":
    model=EmotionTransformer()
    dummy=torch.randn(8,128)
    out=model(dummy)
    print(f"Output shape:{out.shape}")
    print(model)