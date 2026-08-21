import torch
import torch.nn as nn

# Fake data: just to demonstrate the mechanics, not real data
X = torch.randn(100, 3)          # 100 samples, 3 features each
y = torch.randn(100, 1)          # 100 target values

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)  # one layer: 3 inputs -> 1 output

    def forward(self, x):
        return self.linear(x)

model = TinyModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(5):
    predictions = model(X)              # forward pass
    loss = loss_fn(predictions, y)      # compute loss

    optimizer.zero_grad()               # clear old gradients
    loss.backward()                     # backward pass - compute new gradients
    optimizer.step()                    # update weights using those gradients

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")