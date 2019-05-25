import pickle
import numpy as np
import torch
import torch.utils.data

# Load the dataset ...
#  You will need to seperately download or generate this file
infile = open("/Users/raymondfeng/Downloads/RML2016.10a/RML2016.10a_dict.pkl", "rb")
Xd = pickle.load(infile,encoding='latin1')
infile.close()
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)

# Partition the data
#  into training and test sets of the form we can train/test on 
#  while keeping SNR and Mod labels handy for each
np.random.seed(2016)
n_examples = X.shape[0]
n_train = int(n_examples * 0.5)
train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]
def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1
Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

train = torch.utils.data.TensorDataset(torch.from_numpy(X_train),torch.from_numpy(Y_train))
test = torch.utils.data.TensorDataset(torch.from_numpy(X_test),torch.from_numpy(Y_test))

trainloader = torch.utils.data.DataLoader(train, batch_size=4,shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test, batch_size=4,shuffle=False, num_workers=2)

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pad = nn.ZeroPad2d((2, 2, 0, 0))
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(1,3), stride = 1, padding = 0)
        self.conv2 = nn.Conv2d(256, 80, kernel_size=(2,3), stride = 1, padding = 0)
        self.fc1 = nn.Linear(10560, 256)
        self.fc2 = nn.Linear(256, 11)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout2d(p=0.5)
        self.activation = nn.Softmax()

    def forward(self, x):
        x = self.pad(x)
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.pad(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.view(4, 1, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(4, 11)
        x = self.activation(x)
        return x


net = Net()
print(net)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        labels = torch.tensor(labels, dtype=torch.long)
        loss = criterion(outputs, torch.from_numpy(np.array([np.argmax(label) for label in labels])))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
        
#         break

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        labels = torch.from_numpy(np.array([np.argmax(label) for label in labels]))
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))