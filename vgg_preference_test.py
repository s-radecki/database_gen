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
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score

DATASET = "5s_preference_dataset_3"
CSV_FILE = "5s_preference_images.csv"
FULL_MODEL_PATH = "vgg_models/" + "preference/" + "vgg_T_5_E_10_LR_001_M9_TVT_72_18_10__3.pt"
INF_MODEL_PATH = "vgg_models/" + "preference/" + "vgg_T_5_E_10_LR_001_M9_TVT_72_18_10__3_INF.pt"

# validation function
def test(model, test_dataloader, device):
    model.eval()
    running_loss = 0.0
    val_running_correct = 0
    # Initialize the prediction and label lists(tensors)
    predlist=torch.zeros(0,dtype=torch.long, device=device)
    lbllist=torch.zeros(0,dtype=torch.long, device=device)
    for int, data in enumerate(test_dataloader):
        data, target = data['data'].to(device), data['label'].to(device)
        output = model(data)
        loss = criterion(output, target)

        running_loss += loss.item()
        # _, preds = torch.max(output.data, 1, keepdim=True)
        # val_running_correct += (preds == target).sum().item()

        # Append batch prediction results
        _, preds = torch.max(output.data, 1)
        predlist=torch.cat([predlist,preds.view(-1).cpu()])
        # get index of label from target tensors and store in a tensor
        temp = torch.nonzero(target, as_tuple=True)[1]
        lbllist=torch.cat([lbllist,temp])

    test_loss = running_loss/len(test_dataloader.dataset)
    # val_accuracy = 100. * val_running_correct/len(test_dataloader.dataset)
    test_accuracy = 100. * (predlist == lbllist).sum()/len(test_dataloader.dataset)
    # Confusion matrix
    conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
    print(conf_mat)

    # Per-class accuracy
    class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
    print(class_accuracy)

    # Classification report
    report = classification_report(lbllist.numpy(), predlist.numpy())
    print(report)

    # F-1 score
    print("Overall F1-Score: ", f1_score(lbllist.numpy(), predlist.numpy(), average='micro'))

    # Precision
    print("Overall Precision: ", precision_score(lbllist.numpy(), predlist.numpy(), average='micro'))

    # Recall
    print("Overall Recall: ", recall_score(lbllist.numpy(), predlist.numpy(), average='micro'))

    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}')
    # print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {class_accuracy:.2f}')
    return test_loss, test_accuracy


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

# Load test data
test_set = custom_dataset.CustomDataset(DATASET, CSV_FILE, 'test')
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

# Load full model
vgg16 = torch.load(FULL_MODEL_PATH)
print(vgg16)
# loss function
criterion = nn.BCEWithLogitsLoss()

start = time.time()
test_loss, test_accuracy = test(vgg16, test_loader, mps_device)

end = time.time()

print((end-start)/60, 'minutes')