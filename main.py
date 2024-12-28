import torch
import torch.optim as optim
import torch.nn as nn
from cnn_model import CNN, save_weights, load_weights
from data_loader import load_data

def train(model, train_loader, epochs, learning_rate, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            try:
                images, labels = images.to(device), labels.to(device)
                images = images.squeeze(1) 
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            except IndexError as e:
                print(f"Skipping batch due to IndexError: {e}")
        print(f"Epoch {epoch + 1}/{epochs} completed.")

def evaluate(model, test_loader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.squeeze(1)  
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Model accuracy: {accuracy}%")

def main():
    train_X_path = "processed_data/train/X.bin"
    train_y_path = "processed_data/train/y.bin"
    test_X_path = "processed_data/test/X.bin"
    test_y_path = "processed_data/test/Y.bin"

    batch_size = 500
    train_loader = load_data(train_X_path, train_y_path, batch_size)
    test_loader = load_data(test_X_path, test_y_path, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN()

    train(model, train_loader, epochs=50, learning_rate=0.0001, device=device)
    save_weights(model, "cnn_weights.bin")

    print("Model training completed.")

    load_weights(model, "cnn_weights.bin")
    evaluate(model, test_loader, device)

if __name__ == "__main__":
    main()
