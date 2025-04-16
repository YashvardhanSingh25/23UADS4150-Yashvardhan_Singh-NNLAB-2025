#Q7.Write a Program to retrain a pretrained imagenet model to classify a medical image dataset.
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Paths and hyperparameters
base_dir = 'Dataset'  # Should contain CT_COVID and CT_NonCOVID folders
img_size = 224
batch_size = 32
epochs = 20
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# Data Transforms (like ImageDataGenerator in Keras)
train_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),  # Resize images to 224x224
    transforms.RandomHorizontalFlip(),        # Randomly flip images horizontally for data augmentation
    transforms.RandomRotation(15),            # Randomly rotate images up to 15 degrees
    transforms.ToTensor(),                    # Convert image to PyTorch Tensor
    transforms.Normalize([0.485, 0.456, 0.406],  # Normalize using ImageNet mean and std
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load the entire dataset from directory
train_dataset = datasets.ImageFolder(os.path.join(base_dir), transform=train_transforms)

# Split dataset into training and validation
val_size = int(0.2 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# Create DataLoaders for batch loading
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Load pre-trained VGG16 model
base_model = models.vgg16(pretrained=True)

# Freeze convolutional layers (optional fine-tuning of last conv block)
for param in base_model.features.parameters():
    param.requires_grad = False

# Unfreeze last convolutional block (to fine-tune it)
for param in base_model.features[24:].parameters():
    param.requires_grad = True

# Replace the default classifier to suit binary classification
base_model.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 128),  # Flattened feature vector input
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(128, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(256, 1),      # Single output neuron for binary classification
    nn.Sigmoid()            # Sigmoid to output probability between 0 and 1
)

model = base_model.to(device)

# Loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

# Initialize lists to store metrics
train_acc_list, val_acc_list, train_loss_list, val_loss_list = [], [], [], []

# Training Loop
for epoch in range(epochs):
    model.train()
    train_loss, correct, total = 0, 0, 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)  # BCELoss expects float and shape (batch, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        predicted = (outputs > 0.5).float()  # Convert probability to binary prediction
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss /= total
    train_acc = correct / total
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)

    # Validation Loop
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= total
    val_acc = correct / total
    val_acc_list.append(val_acc)
    val_loss_list.append(val_loss)

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Save the trained model weights
torch.save(model.state_dict(), 'covid_classifier_vgg16.pt')

# Plot Accuracy and Loss curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_acc_list, label='Train Acc')
plt.plot(val_acc_list, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Final Evaluation on the Validation Set
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(images)
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total
print(f"\n Final Test Accuracy on Validation Set: {test_acc:.4f}")
