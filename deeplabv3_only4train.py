import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
import torchvision.models.segmentation as segmentation
import torch.optim as optim
 
#Custom the class of Dataset
class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_transform=None ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")#Grally mode

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
            mask = torch.squeeze(mask, 0) #make sure that the shape of mask is [batch_size, height, width]

        return image, mask
    

#Define data transform
image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((512, 512), interpolation=Image.NEARSET),
])

#Load dataset
train_dataset = CustomDataset(
    image_dir='WS_seg/WildScenes_demo/image',
    mask_dir='WS_seg/WildScenes_demo/label',
    transform=image_transform,
    target_transform=mask_transform
)

#Fill the last batch
def pad_last_batch(dataset, batch_size):
    dataset_size = len(dataset)
    pad_size = (batch_size - dataset_size % batch_size) % batch_size
    indices = list(range(dataset_size))
    indices += indices[:pad_size]
    return SubsetRandomSampler(indices)

batch_size = 4
sampler = pad_last_batch(train_dataset, batch_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)

#Define the model, loss function and optimizer
num_classes = 15+1 #include background
model = segmentation.deeplabv3_resnet50(pretrained=True)
model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#train the model
num_epochs = 25

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        optputs = model(images)['out']
        loss = criterion(optputs, masks.long())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

print("Finished Training")


#Save model
torch.save(model.state_dict(), 'deeplabv3_custom.pth')