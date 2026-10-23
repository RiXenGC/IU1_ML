from torch import nn

class WeatherLSTM(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size=32,
                 num_layers=1, 
                 head_type="mlp"
                ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        
        if head_type == "linear":
            self.head = nn.Sequential(nn.Linear(hidden_size, 1))
        elif head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )
        else:
            raise ValueError("head_type должен быть 'linear' или 'mlp'.")

    def forward(self, x):
        sequence, _ = self.lstm(x)
        return self.head(sequence[:, -1, :]).squeeze(-1)
