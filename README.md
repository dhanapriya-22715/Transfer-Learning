# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
1. Develop a binary classification model using a pretrained VGG19 to distinguish between defected and non-defected capacitors by modifying the last layer to a single neuron.  
2. Train the model on a dataset containing images of various defected and non-defected capacitors to improve defect detection accuracy.  
3. Optimize and evaluate the model to ensure reliable classification for capacitor quality assessment in manufacturing.

## DESIGN STEPS
### STEP 1:
Collect and preprocess the dataset containing images of defected and non-defected capacitors.

### STEP 2:
Split the dataset into training, validation, and test sets.

### STEP 3:
Load the pretrained VGG19 model with weights from ImageNet.

### STEP 4:
Remove the original fully connected (FC) layers and replace the last layer with a single neuron (1 output) with a Sigmoid activation function for binary classification.

### STEP 5:
Train the model using binary cross-entropy loss function and Adam optimizer.

### STEP 6:
Evaluate the model with test data loader and intepret the evaluation metrics such as confusion matrix and classification report.


## PROGRAM
Include your code here
```python
# Load Pretrained Model and Modify for Transfer Learning
# Load a pre-trained VGG19 model
from torchvision.models import VGG19_Weights
model=models.vgg19(weights=VGG19_Weights.DEFAULT)

for param in model.features.parameters():
    param.requires_grad = False  # Freeze feature extractor layers


# Modify the final fully connected layer to match the dataset classes
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, 1)

# Include the Loss function and optimizer

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Train the model
def train_model(model, train_loader, test_loader, num_epochs=10):
    train_losses = []
    val_losses = []

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)  # 🔧 FIX

            optimizer.zero_grad()
            outputs = model(images)          # (B, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device).float().unsqueeze(1)  # 🔧 FIX

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(
            f'Epoch [{epoch+1}/{num_epochs}], '
            f'Train Loss: {train_losses[-1]:.4f}, '
            f'Validation Loss: {val_losses[-1]:.4f}'
        )

    # Plot losses
    print("Name: Dhanappriya S")
    print("Register Number: 212224230056")

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
<img width="779" height="414" alt="image" src="https://github.com/user-attachments/assets/d9469e55-8c65-4f1c-baed-efb9e1f22e33" />

<img width="763" height="620" alt="image" src="https://github.com/user-attachments/assets/74b1804c-3fa2-4e41-a726-1f6b9aa30aa2" />



### Confusion Matrix
<img width="808" height="678" alt="image" src="https://github.com/user-attachments/assets/70342fb1-9672-4f5f-b67e-5ffb01f3b696" />


### Classification Report
<img width="775" height="289" alt="image" src="https://github.com/user-attachments/assets/6643a4a2-1aaf-4b1e-992e-790ba63094df" />


### New Sample Prediction
<img width="714" height="514" alt="image" src="https://github.com/user-attachments/assets/be3164d1-a28f-49f1-9c55-7252e6dde8a4" />
<img width="564" height="521" alt="image" src="https://github.com/user-attachments/assets/e5206ba7-e5f3-460b-a80a-a9429f69dfe2" />

## RESULT
The VGG-19 model was successfully trained and optimized to classify defected and non-defected capacitors.
