import model
import feature_extraction
import model
from tqdm import tqdm
import torch

filename = 'data/ADABUSD_5min.csv' # test for now
train_df, test_df, train_dist = feature_extraction.feature_dfs_from_klines(filename, train_frac=0.9)
target_name = 'next_return'
feature_names = [col for col in train_df.columns if col != target_name]
sequence_length = 10


train_set = feature_extraction.SequenceDataset(
    df=train_df,
    target=target_name,
    features=feature_names,
    sequence_length=sequence_length
)

test_set = feature_extraction.SequenceDataset(
    df=test_df,
    target=target_name,
    features=feature_names,
    sequence_length=sequence_length
)
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
X, y = next(iter(train_loader))

print('Feature shape:', X.shape, ", Target shape:", y.shape)

n_epochs = 10
lr = 5e-5
input_dim = 9
hidden_units = 32
num_layers = 2
output_dim = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

network = model.LSTM(input_dim, hidden_units, num_layers, output_dim)
network = network.to(device)
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(network.parameters(), lr=lr)

# Training

for epoch in range(n_epochs):
    print('epoch: ', epoch)
    # train
    network.train()
    for X, y in tqdm(train_loader):
        hidden = network.init_hidden(batch_size=len(y), device=device)
        X = X.to(device)
        y = y.to(device)
        y_hat = network(X, hidden)
        l = loss(y_hat, y)
        
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    
    # test
    network.eval()
    with torch.no_grad():
        n_batches = len(test_loader)
        test_loss = 0
        for X, y in tqdm(test_loader):
            hidden = network.init_hidden(batch_size=len(y), device=device)
            X = X.to(device)
            y = y.to(device)
            y_hat = network(X, hidden)
            test_loss += loss(y_hat, y)
        
        test_loss = test_loss/n_batches

        print('test loss:', test_loss.item())


# TODO: Plot some predictions (rescaled back to real values)