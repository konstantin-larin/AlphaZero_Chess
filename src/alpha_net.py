#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import datetime
import numpy as np

class board_data(Dataset):
    def __init__(self, dataset, seed=None): # dataset = np.array of (s, p, v)                
        if seed is not None:
            np.random.seed(seed)
        indices = np.random.permutation(len(dataset['s'])) # shuffle data
        self.X = torch.from_numpy(np.array(dataset['s'], dtype=np.float32)[indices])               
        self.y_p = torch.from_numpy(np.array(dataset['p'], dtype=np.float32)[indices])      
        self.y_v = torch.tensor(dataset['v'], dtype=torch.int8)[indices]
    
    def __len__(self):        
        return len(self.X)
    
    def __getitem__(self,idx):        
        return self.X[idx].permute(2,0,1), self.y_p[idx], self.y_v[idx]

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.action_size = 8*8*73 #4672
        self.conv1 = nn.Conv2d(22, 256, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

    def forward(self, s):
        s = s.view(-1, 22, 8, 8)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        return s

class ResBlock(nn.Module):
    def __init__(self, inplanes=256, planes=256, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out
    
class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(256, 1, kernel_size=1) # value head
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8*8, 64)
        self.fc2 = nn.Linear(64, 1)
        
        self.conv1 = nn.Conv2d(256, 128, kernel_size=1) # policy head
        self.bn1 = nn.BatchNorm2d(128)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(8*8*128, 8*8*73)
    
    def forward(self,s):
        v = F.relu(self.bn(self.conv(s))) # value head
        v = v.view(-1, 8*8)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = F.tanh(self.fc2(v))
        
        p = F.relu(self.bn1(self.conv1(s))) # policy head
        p = p.view(-1, 8*8*128)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v
    
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv = ConvBlock()
        for block in range(19):
            setattr(self, "res_%i" % block,ResBlock())
        self.outblock = OutBlock()
    
    def forward(self,s):
        s = self.conv(s)
        for block in range(19):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s
        

class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum((-policy* 
                                (1e-6 + y_policy.float()).float().log()), 1)
        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error
    

def train(net, train_data, val_data=None, epochs=20, seed=0, save_path='./model_data/'):
    torch.manual_seed(seed)
    cuda = torch.cuda.is_available()
    net.train()

    criterion = AlphaLoss()
    optimizer = optim.Adam(net.parameters(), lr=3e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 400], gamma=0.2)

    pin_memory = True if cuda else False
    train_set = board_data(train_data, seed=seed)    
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0, pin_memory=pin_memory)
    if val_data is not None:
        val_set = board_data(val_data, seed=seed)    
        val_loader = DataLoader(val_set, batch_size=64, shuffle=True, num_workers=0, pin_memory=pin_memory)

    losses_per_epoch = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        
        for i, (state, policy, value) in enumerate(train_loader):            
            if cuda:
                state, policy, value = state.cuda(), policy.cuda(), value.cuda()

            optimizer.zero_grad()
            policy_pred, value_pred = net(state)
            loss = criterion(value_pred[:, 0], value, policy_pred, policy)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

            if i % 10 == 0:
                print(f"[Epoch {epoch+1}/{epochs}] Batch {i}: loss = {loss.item():.4f}")
        
        if batches > 0:
            avg_loss = epoch_loss / batches
            losses_per_epoch.append(avg_loss)
            print(f"Epoch {epoch+1}: avg_loss = {avg_loss:.4f}")
        else:
            print("No batches in this epoch — skipping.")
            continue

        if val_data is not None:
            with torch.no_grad():
                val_loss = 0.0
                correct_policy = 0
                total_samples = 0
                val_batches = 0

                for state, policy, value in val_loader:
                    if cuda:
                        state, policy, value = state.cuda(), policy.cuda(), value.cuda()
                    policy_pred, value_pred = net(state)
                    loss = criterion(value_pred[:, 0], value, policy_pred, policy)
                    val_loss += loss.item()
                    val_batches += 1

                    # Policy accuracy
                    pred_moves = policy_pred.argmax(dim=1)
                    true_moves = policy.argmax(dim=1)
                    correct_policy += (pred_moves == true_moves).sum().item()
                    total_samples += state.size(0)

            val_loss /= val_batches
            accuracy = correct_policy / total_samples if total_samples > 0 else 0.0
            print(f"Validation loss: {val_loss:.4f}, Moves accuracy: {accuracy:.4f}")

        scheduler.step()
        
        if len(losses_per_epoch) > 5 and abs(losses_per_epoch[-1] - losses_per_epoch[-5]) < 1e-3:
            print("Early stopping: loss plateau.")
            break        

    plt.figure()
    plt.plot(range(1, len(losses_per_epoch)+1), losses_per_epoch)
    plt.xlabel("Epoch")
    plt.ylabel("Avg Loss")
    plt.title("Training Loss per Epoch")
    plt.savefig(os.path.join(save_path, f"Loss_vs_Epoch.png"))
    print("Finished Training")



def test(net, test_data, seed=0):
    torch.manual_seed(seed)
    cuda = torch.cuda.is_available()
    net.eval()

    criterion = AlphaLoss()
    pin_memory = True if cuda else False
    test_set = board_data(test_data, seed=seed)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0, pin_memory=pin_memory)

    total_loss = 0.0
    correct_policy = 0
    total_samples = 0
    batches = 0

    with torch.no_grad():
        for state, policy, value in test_loader:
            if cuda:
                state, policy, value = state.cuda(), policy.cuda(), value.cuda()
            
            policy_pred, value_pred = net(state)
            loss = criterion(value_pred[:, 0], value, policy_pred, policy)
            total_loss += loss.item()
            batches += 1

            # Accuracy для policy
            pred_moves = policy_pred.argmax(dim=1)
            true_moves = policy.argmax(dim=1)
            correct_policy += (pred_moves == true_moves).sum().item()
            total_samples += state.size(0)

    avg_loss = total_loss / batches if batches > 0 else float("inf")
    accuracy = correct_policy / total_samples if total_samples > 0 else 0.0

    print(f"Test loss: {avg_loss:.4f}, Moves accuracy: {accuracy:.4f}")
    return avg_loss, accuracy



# def predict(net, board):
        