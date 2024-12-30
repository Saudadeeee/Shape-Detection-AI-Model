import torch
import torch.optim as optim
import torch.nn as nn
from cnn_model import CNN, save_weights_binary, load_weights_binary
from data_loader import load_data
from torch.optim.lr_scheduler import StepLR
import numpy as np

def train(model, train_loader, epochs, learning_rate, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

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
        scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs} completed. Loss: {loss.item()}")

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

def predict_image(model, image_path, device):
    model.to(device)
    model.eval()
    image = prepare_image(image_path)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)  # Fix the shape of the image tensor
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

def predict_image_from_binary(model, binary_path, device):
    model.to(device)
    model.eval()
    image = np.fromfile(binary_path, dtype=np.float32)
    if image.size == 64 * 64 * 2:
        image = image[:64 * 64]  # Fix the size if it is doubled
    if image.size != 64 * 64:
        raise ValueError(f"Expected binary file to contain 4096 elements, but got {image.size}")
    image = image.reshape(1, 1, 64, 64)  # Fix the shape to [1, 1, 64, 64]
    image = torch.tensor(image, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

def main():
    train_X_path = "processed_data/train/X.bin"
    train_y_path = "processed_data/train/y.bin"
    test_X_path = "processed_data/test/X.bin"
    test_y_path = "processed_data/test/Y.bin"

    batch_size = 1000
    train_loader = load_data(train_X_path, train_y_path, batch_size)
    test_loader = load_data(test_X_path, test_y_path, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN()

    print("Training model...")
    train(model, train_loader, epochs=30, learning_rate=0.0001, device=device)
    save_weights_binary(model, "cnn_weights.bin")
    print("Model training completed.")

    load_weights_binary(model, "cnn_weights.bin")
    print("Evaluating model...")
    evaluate(model, test_loader, device)

    load_weights_binary(model, "cnn_weights.bin")
    binary_image_path = 'd:/Code/SourceCode/CNN_ModelAI/test.bin'
    predicted_class = predict_image_from_binary(model, binary_image_path, device)
    label_map = {0: "circle", 1: "square", 2: "star", 3: "triangle"}
    print(f"The predicted class for {binary_image_path} is: {label_map[predicted_class]}")

if __name__ == "__main__":
    main()
