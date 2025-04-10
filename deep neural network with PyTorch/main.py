import torch
from data import *
from torch.utils.data import DataLoader, TensorDataset
from NeuralNet import *

# Convert to tensors - use y_train and y_test (not X_train for targets)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.argmax(axis=1), dtype=torch.long)  # Convert one-hot to class indices
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.argmax(axis=1), dtype=torch.long)    # Convert one-hot to class indices

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = NeuralNet()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Iteracja numer: [{epoch+1}/{num_epochs}], Błąd sieci: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, dim=1)
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f'Dokładność modeli na podstawie danych testowych: poprawnych: {accuracy * 100:.2f}%')

torch.save(model, 'classifier_model.pth')
