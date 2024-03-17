import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from gesture_dataset import GestureDataset
from model import Prev_Net, LeNet5, VGGNet16, ResNet
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

epochs = 200
best_acc_ratio = 0.
total_data_path = '/home/jabblee/Desktop/CRC_collections/CRC_update/2023_Gatherings/'

# Datasets and DataLoaders for train and test
train_dataset = GestureDataset(data_path = total_data_path)
test_dataset = GestureDataset(data_path = total_data_path, train = False)

train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)

# Model
# model = Prev_Net().cuda()
model = ResNet().cuda()
# model = VGGNet16().cuda()

# Loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# Containers
train_loss_container = []
test_loss_container = []
train_accuracy_container = []
test_accuracy_container = []

test_best_epoch = 0
for epoch in tqdm(range(epochs)):
    model.train()
    
    train_loss = 0
    train_acc_point = 0
    test_acc_point = 0
    total_train_batch = len(train_loader)
    
    for train_data in train_loader:
        optimizer.zero_grad()
        
        train_data['gesture_data'] = train_data['gesture_data'].cuda().float()
        train_data['class_label'] = train_data['class_label'].cuda().long()
        
        pred = model(train_data['gesture_data'])
        loss = criterion(pred, train_data['class_label'].long())
        
        loss.backward()
        optimizer.step()
        
        train_correct_prediction = torch.argmax(pred, 1) == train_data['class_label']
        train_loss += loss.item() / total_train_batch

        if train_correct_prediction.any():
            train_acc_point += 1
    
    train_acc_ratio = train_acc_point / total_train_batch
    train_loss_container.append(train_loss)
    # train_accuracy_container.append(train_acc_ratio)

    print('* Epoch : ', '%04d' % (epoch+1), 'Loss : ', '{:.9f}'.format(train_loss))

    model.eval()
    
    test_loss = 0
    test_acc_point = 0

    total_test_batch = len(test_loader)
    validation_arr = []
    validation_arr = np.zeros((23, 23))
    
    with torch.no_grad():
        for test_data in test_loader:
            test_data['gesture_data'] = test_data['gesture_data'].cuda().float()
            test_data['class_label'] = test_data['class_label'].cuda().long()
            
            pred = model(test_data['gesture_data'])
            
            loss = criterion(pred, test_data['class_label'].long())
            
            test_correct_prediction = torch.argmax(pred, 1) == test_data['class_label']
            validation_arr[test_data['class_label'].item()][torch.argmax(pred, 1)] += 1

            test_loss += loss.item() / total_test_batch
            
            if test_correct_prediction.any():
                test_acc_point += 1
                
    test_acc_ratio = test_acc_point / total_test_batch
    if test_acc_ratio > best_acc_ratio:
        best_acc_ratio = test_acc_ratio
        test_best_epoch = epoch
    
    test_loss_container.append(test_loss)
    test_accuracy_container.append(test_acc_ratio)
    
    
    print('Accuracy Ratio : {}%'.format(test_acc_ratio))

# print('Best Accuracy :', best_acc_ratio)

PATH = '/home/jabblee/Desktop/CRC_collections/CRC/output/' + str(test_best_epoch) + '_state_dict.pt'
torch.save(model.state_dict(), PATH)
plt.title('ResNet Best Test Accuracy : ' + str(best_acc_ratio))
plt.plot(train_loss_container, label = 'train_loss')
plt.plot(test_loss_container, label = 'test_loss')
plt.plot(test_accuracy_container, label = 'test_accuracy')
plt.legend()
plt.show()