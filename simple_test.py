import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from grokadamw import GrokAdamw8bit

# Set random seed for reproducibility
torch.manual_seed(42)

# Create a simple dataset
X = torch.randn(1000, 10)
y = torch.randn(1000, 1)
train_size = 800
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Define the model
model = nn.Linear(10, 1)

# Define your grokking signal function(s)
def grokking_signal_fn(training_loss: float, validation_loss: float) -> float:
    if training_loss == 0:
        return 0.0  # Avoid division by zero
    return (validation_loss - training_loss) / training_loss

# Initialize GrokAdamW optimizer
lr = torch.tensor(0.001)
optimizer = GrokAdamw8bit(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    # Calculate grokking signal
    grokking_signal = grokking_signal_fn(train_loss, val_loss)
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, ")

print("Training complete!")