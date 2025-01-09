import torch
import torch.optim as optim
import torch.nn as nn
from cnn_model import CNN
from data_loader import load_data
from torch.optim.lr_scheduler import StepLR
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train(model, train_loader, val_loader, epochs, learning_rate, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            try:
                images, labels = images.to(device), labels.to(device)
                images = images.squeeze(1)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            except IndexError as e:
                print(f"Skipping batch due to IndexError: {e}")
        scheduler.step()
        epoch_loss = running_loss / (i + 1)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs} completed. Training Loss: {epoch_loss}")

        # Evaluate on validation set
        val_loss = evaluate_loss(model, val_loader, device, criterion)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}/{epochs} completed. Validation Loss: {val_loss}")

    # Plot training and validation loss
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.show()

def evaluate_loss(model, data_loader, device, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.squeeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def evaluate(model, test_loader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.squeeze(1)  
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    accuracy = 100 * correct / total
    precision = 100 * precision_score(all_labels, all_predictions, average='weighted')
    recall = 100 * recall_score(all_labels, all_predictions, average='weighted')
    f1 = 100 * f1_score(all_labels, all_predictions, average='weighted')
    avg_loss = total_loss / len(test_loader)
    print(f"Model accuracy: {accuracy}%")
    print(f"Precision: {precision}%")
    print(f"Recall: {recall}%")
    print(f"F1 Score: {f1}%")
    print(f"Average Loss: {avg_loss}")

    # Plot evaluation metrics
    metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1, 'Loss': avg_loss}
    plt.figure()
    plt.bar(metrics.keys(), metrics.values())
    plt.title('Evaluation Metrics')
    plt.show()

    # Compute and display confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def predict_image(model, image_path, device):
    model.to(device)
    model.eval()
    image = prepare_image(image_path)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)  
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

def predict_image_from_binary(model, binary_path, device):
    model.to(device)
    model.eval()
    image = np.fromfile(binary_path, dtype=np.float32)
    if image.size == 64 * 64 * 2:
        image = image[:64 * 64]  
    image = image.reshape(1, 1, 64, 64)  
    image = torch.tensor(image, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

def export_hardcoded_weights(model):
    print("Copy the following lines into cnn_model.py inside __init__() with torch.no_grad():\n")
    sd = model.state_dict()
    for k, v in sd.items():
        if "weight" in k or "bias" in k:
            arr = v.flatten().tolist()
            print(f"self.{k}.data = torch.tensor({arr}, dtype=torch.float32).view(*self.{k}.data.shape)")

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
    train(model, train_loader, test_loader, epochs=30, learning_rate=0.0001, device=device)
    print("Model training completed.")

    # Export weights after training
    export_hardcoded_weights(model)
    print("Copy the above lines into cnn_model.py inside __init__() with torch.no_grad()")

    print("Evaluating model...")
    evaluate(model, test_loader, device)

    binary_image_path = 'd:/Code/SourceCode/CNN_ModelAI/test.bin'
    predicted_class = predict_image_from_binary(model, binary_image_path, device)
    label_map = {0: "circle", 1: "square", 2: "star", 3: "triangle"}
    print(f"The predicted class for {binary_image_path} is: {label_map[predicted_class]}")

if __name__ == "__main__":
    main()
