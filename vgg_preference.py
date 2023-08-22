import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models
import custom_dataset
from sklearn.metrics import confusion_matrix, classification_report

DATASET = "5s_preference_dataset_3"
CSV_FILE = "5s_preference_images.csv"
EPOCHS = 10
FULL_MODEL_PATH = "vgg_models/" + "preference/" + "vgg_T_5_E_10_LR_001_M9_TVT_72_18_10__3.pt"
INF_MODEL_PATH = "vgg_models/" + "preference/" + "vgg_T_5_E_10_LR_001_M9_TVT_72_18_10__3_INF.pt"

# validation function
def validate(model, test_dataloader, device):
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    # Initialize the prediction and label lists(tensors)
    predlist=torch.zeros(0,dtype=torch.long, device=device)
    lbllist=torch.zeros(0,dtype=torch.long, device=device)
    for int, data in enumerate(test_dataloader):
        data, target = data['data'].to(device), data['label'].to(device)
        output = model(data)
        loss = criterion(output, target)

        val_running_loss += loss.item()
        # _, preds = torch.max(output.data, 1, keepdim=True)
        # val_running_correct += (preds == target).sum().item()

        # Append batch prediction results
        _, preds = torch.max(output.data, 1)
        predlist=torch.cat([predlist,preds.view(-1).cpu()])
        # get index of label from target tensors and store in a tensor
        temp = torch.nonzero(target, as_tuple=True)[1]
        lbllist=torch.cat([lbllist,temp])

    val_loss = val_running_loss/len(test_dataloader.dataset)
    # val_accuracy = 100. * val_running_correct/len(test_dataloader.dataset)
    val_accuracy = 100. * (predlist == lbllist).sum()/len(test_dataloader.dataset)
    # Confusion matrix
    conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
    print(conf_mat)

    # Per-class accuracy
    class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
    print(class_accuracy)

    # Classification report
    report = classification_report(lbllist.numpy(), predlist.numpy())
    print(report)

    print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_accuracy:.2f}')
    # print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {class_accuracy:.2f}')
    return val_loss, val_accuracy


# training function
def fit(model, train_dataloader, device):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    # Initialize the prediction and label lists(tensors)
    predlist=torch.zeros(0,dtype=torch.long, device=device)
    lbllist=torch.zeros(0,dtype=torch.long, device=device)
    for i, data in enumerate(train_dataloader):
        data, target = data['data'].to(device), data['label'].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_running_loss += loss.item()
        # _, preds = torch.max(output.data, 1, keepdim=True)
        _, preds = torch.max(output.data, 1)
        # train_running_correct += (preds == target).sum().item() # preds.eq(target.data).sum().item()  # (preds == target).sum().item()
        # Append batch prediction results
        predlist=torch.cat([predlist,preds.view(-1).cpu()])
        # get index of label from target tensors and store in a tensor
        temp = torch.nonzero(target, as_tuple=True)[1]
        lbllist=torch.cat([lbllist,temp])
        loss.backward()
        optimizer.step()
    train_loss = train_running_loss/len(train_dataloader.dataset)
    # train_accuracy = 100. * train_running_correct/len(train_dataloader.dataset)
    train_accuracy = 100. * (predlist == lbllist).sum()/len(train_dataloader.dataset)
    # Confusion matrix
    conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
    print(conf_mat)

    # Per-class accuracy
    class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
    print(class_accuracy)

    # Classification report
    report = classification_report(lbllist.numpy(), predlist.numpy())
    print(report)

    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}')
    # print(f'Train Loss: {train_loss:.4f},') # Train Acc: {class_accuracy:.2f}')

    return train_loss, train_accuracy


# mps - enables GPU for macOS devices
# check if mps is available
# if torch.backends.mps.is_available():
#     mps_device = torch.device("mps")
#     # x = torch.ones(1, device=mps_device)
#     x = torch.rand(size=(3, 4)).to(mps_device)
#     print(x)
# else:
#     print("MPS device not found.")

# No suport for adaptive pooling on MPS
# https://github.com/pytorch/pytorch/issues/96056
mps_device = torch.device("cpu")
print(mps_device)

# create and load datasets
# - https://pytorch.org/docs/stable/data.html
# - https://debuggercafe.com/transfer-learning-with-pytorch/
train_set = custom_dataset.CustomDataset(DATASET, CSV_FILE, 'train')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

test_set = custom_dataset.CustomDataset(DATASET, CSV_FILE, 'test')
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

val_set = custom_dataset.CustomDataset(DATASET, CSV_FILE, 'val')
val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=True)

vgg16 = models.vgg16(weights='DEFAULT')
vgg16.to(mps_device)
# print(vgg16)


# change the number of classes to 2
vgg16.classifier[6].out_features = 2
# freeze convolution weights
for param in vgg16.features.parameters():
    param.requires_grad = False

# https://www.kaggle.com/code/carloalbertobarbano/vgg16-transfer-learning-pytorch
# Newly created modules have require_grad=True by default
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, 2)]) # Add our layer with 2 outputs
vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
print(vgg16)

# optimizer
optimizer = optim.SGD(vgg16.classifier.parameters(), lr=0.001, momentum=0.9)
# loss function - Binray cross entropy
criterion = nn.BCEWithLogitsLoss()

train_loss , train_accuracy = [], []
val_loss , val_accuracy = [], []
start = time.time()
for epoch in range(EPOCHS):
    train_epoch_loss, train_epoch_accuracy = fit(vgg16, train_loader, mps_device)
    val_epoch_loss, val_epoch_accuracy = validate(vgg16, val_loader, mps_device)
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
end = time.time()

# https://pytorch.org/tutorials/beginner/saving_loading_models.html
# Save full model
torch.save(vgg16, FULL_MODEL_PATH)

# Save for inference
torch.save(vgg16.state_dict(), INF_MODEL_PATH)

print((end-start)/60, 'minutes')