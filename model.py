import torch

class LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_units, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.lstm = torch.nn.LSTM(
            input_size=self.input_dim, 
            hidden_size=self.hidden_units, 
            num_layers=self.num_layers, 
            batch_first=True
        )

        self.fc = torch.nn.Linear(in_features=self.hidden_units, out_features=self.output_dim)

    def init_hidden(self, batch_size, device='cpu'):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_size, self.hidden_units).to(device),
                weight.new_zeros(self.num_layers, batch_size, self.hidden_units).to(device))

    def forward(self, x, hidden): # have to put hidden on device
        _, (hn, _) = self.lstm(x, (hidden))
        out = self.fc(hn[0]).flatten()
        return out
