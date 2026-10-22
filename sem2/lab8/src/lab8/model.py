from torch import nn

class WeatherLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        sequence, _ = self.lstm(x)
        last_state = sequence[:, -1, :]
        prediction = self.head(last_state)
        return prediction.squeeze(-1)
