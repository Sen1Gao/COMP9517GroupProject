import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import numpy as np
import matplotlib.pyplot as plt

# Import custom scripts for classification conversion and data generation
import classification_conversion
import data_generator

# Apply classification conversion (assuming this function is defined in the script)
classification_conversion.convert_to_15_classes('path/to/labels')  # Adjust the path as needed

# Generate data splits with uniform class distribution
data_generator.generate_splits(input_dir='path/to/dataset', output_dir='path/to/splits')  # Adjust paths as needed

# Dataset definition
class WildScenesDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        label = Image.open(os.path.join(self.label_dir, img_name))
        
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        
        return image, label

# Data transformation
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create datasets
train_dataset = WildScenesDataset(image_dir='splits/train/image', label_dir='splits/train/indexLabel', transform=transform)
val_dataset = WildScenesDataset(image_dir='splits/val/image', label_dir='splits/val/indexLabel', transform=transform)
test_dataset = WildScenesDataset(image_dir='splits/test/image', label_dir='splits/test/indexLabel', transform=transform)

# Sampling a subset of the dataset
def sample_dataset(dataset, sample_fraction=0.3):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    sample_size = int(dataset_size * sample_fraction)
    random.shuffle(indices)
    sample_indices = indices[:sample_size]
    return Subset(dataset, sample_indices)

# Sample the datasets (e.g., 30% of the data)
train_dataset = sample_dataset(train_dataset, sample_fraction=0.3)
val_dataset = sample_dataset(val_dataset, sample_fraction=0.3)
test_dataset = sample_dataset(test_dataset, sample_fraction=0.3)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# SE-Block definition
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Modified DeepLabv3 model with SE-Block integration and EfficientNet encoder
class DeepLabv3_SE(smp.DeepLabV3):
    def __init__(self, encoder_name='efficientnet-b4', encoder_weights='imagenet', classes=15, activation=None):
        super(DeepLabv3_SE, self).__init__(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=classes, activation=activation)
        self.se_block = SEBlock(channels=self.encoder.out_channels[-1])  # Adjust channels based on the encoder
    
    def forward(self, x):
        features = self.encoder(x)
        x = self.aspp(features[-1])
        x = self.se_block(x)
        x = self.decoder(x)
        return x

# Initialize model, criterion, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepLabv3_SE(encoder_name='efficientnet-b4', encoder_weights='imagenet', classes=15).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
best_iou = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.long())  # Ensure labels are long type
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

    # Evaluate on validation set
    model.eval()
    iou_scores = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            iou = calculate_iou(preds.cpu().numpy(), labels.cpu().numpy(), num_classes=15)
            iou_scores.append(iou)

    mean_iou = np.mean(iou_scores)
    print(f'Validation Mean IoU: {mean_iou}')

    # Save the best model
    if mean_iou > best_iou:
        best_iou = mean_iou
        torch.save(model.state_dict(), 'best_model.pth')
        print('Model saved!')

# IoU calculation function
def calculate_iou(predictions, labels, num_classes):
    iou_list = []
    for cls in range(num_classes):
        pred_mask = (predictions == cls)
        label_mask = (labels == cls)
        intersection = (pred_mask & label_mask).sum().item()
        union = (pred_mask | label_mask).sum().item()
        if union == 0:
            iou_list.append(float('nan'))
        else:
            iou_list.append(intersection / union)
    return np.nanmean(iou_list)

# Evaluation on test set
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
iou_scores = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        iou = calculate_iou(preds.cpu().numpy(), labels.cpu().numpy(), num_classes=15)
        iou_scores.append(iou)

print(f'Test Mean IoU: {np.mean(iou_scores)}')

# Show representative examples of segmentations
def visualize_segmentation(image, prediction, ground_truth, classes):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())

    plt.subplot(1, 3, 2)
    plt.title('Prediction')
    plt.imshow(prediction.cpu().numpy(), cmap='jet', vmin=0, vmax=classes)

    plt.subplot(1, 3, 3)
    plt.title('Ground Truth')
    plt.imshow(ground_truth.cpu().numpy(), cmap='jet', vmin=0, vmax=classes)

    plt.show()

# Visualize some examples
for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    
    for i in range(images.size(0)):
        visualize_segmentation(images[i], preds[i], labels[i], num_classes=15)
        # Break after a few examples
        break
    break

# Compare with the WildScenes dataset's results
# Assuming we have ground truth IoU scores for comparison (replace with actual values)
wildscenes_iou = [0.75, 0.65, 0.78, 0.80, 0.70, 0.60, 0.85, 0.90, 0.88, 0.55, 0.77, 0.66, 0.79, 0.83, 0.60]
our_iou = np.mean(iou_scores)
print(f'WildScenes Mean IoU: {np.mean(wildscenes_iou)}, Our Mean IoU: {our_iou}')

# Discussing results
print("Our method performed better/worse in the following aspects...")
print("Potential reasons for differences include...")
print("Future research directions could involve...")
