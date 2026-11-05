from torch import nn


class WeatherLSTM(nn.Module):

    def __init__(self, input_size, hidden_size=16, num_layers=1, head_type="linear"):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, 1))

    def forward(self, x):
        # x: (примеры, 30 дней, признаки).
        sequence, _ = self.lstm(x)
        # Представление последнего дня учитывает всё входное окно.
        last_day = sequence[:, -1, :]
        prediction = self.head(last_day)
        # (примеры, 1) -> (примеры,), как у y в prepared.npz.
        return prediction.squeeze(-1)
