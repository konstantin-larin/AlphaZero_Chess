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
import h5py
from tqdm import tqdm
from torch.utils.data import Subset
import mlflow

BOARD_X = 8
BOARD_Y = 8
BOARD_CHANNELS = 22
LEGAL_MOVES = 73
KERNEL_SIZE = 3
PADDING = 1
STRIDE= 1 


class board_data(Dataset):
    def __init__(self, datapath):
        self.f = h5py.File(datapath, 'r')
        self.s = self.f['s']
        self.p = self.f['p']
        self.v = self.f['v']

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.s[idx]).permute(2,0,1).float()
        y_p = torch.from_numpy(self.p[idx])
        y_v = torch.tensor(self.v[idx], dtype=torch.float32)
        return x, y_p, y_v


class ConvBlock(nn.Module):
    def __init__(self, planes, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.action_size = BOARD_X*BOARD_Y*LEGAL_MOVES 
        self.conv1 = nn.Conv2d(BOARD_CHANNELS, planes, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, s):
        s = s.view(-1, BOARD_CHANNELS, BOARD_X, BOARD_Y)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        return s

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride, padding, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=False)
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
    def __init__(self, inplanes, value_hidden_dim, policy_hidden_dim):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, 1, kernel_size=1) # value head
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(BOARD_X*BOARD_Y, value_hidden_dim)
        self.fc2 = nn.Linear(value_hidden_dim, 1)
        
        self.conv1 = nn.Conv2d(inplanes, policy_hidden_dim, kernel_size=1) # policy head
        self.bn1 = nn.BatchNorm2d(policy_hidden_dim)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(BOARD_X*BOARD_Y*policy_hidden_dim, BOARD_X*BOARD_Y*LEGAL_MOVES)
        self.policy_hidden_dim = policy_hidden_dim
    
    def forward(self,s):
        v = F.relu(self.bn(self.conv(s))) # value head
        v = v.view(-1, BOARD_X*BOARD_Y)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = F.tanh(self.fc2(v))
        
        p = F.relu(self.bn1(self.conv1(s))) # policy head
        p = p.view(-1, BOARD_X*BOARD_Y*self.policy_hidden_dim)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v
    
class ChessNet(nn.Module):
    def __init__(self, 
                 res_blocks_num=19,              
                 planes=256,                                                                                                                         
                 value_hidden_dim=64,
                 policy_hidden_dim=128,
                 

                 name='default_chessnet'):
        super(ChessNet, self).__init__()
        self.conv = ConvBlock(planes=planes, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING)
        self.name = name
        self.res_blocks_num = res_blocks_num
        for block in range(res_blocks_num):
            setattr(self, "res_%i" % block,
                    ResBlock(
                        inplanes=planes,
                        planes=planes,
                        kernel_size=KERNEL_SIZE,
                        stride=STRIDE,
                        padding=PADDING,                        
                    ))
        self.outblock = OutBlock(
            inplanes=planes,
            value_hidden_dim=value_hidden_dim,            
            policy_hidden_dim=policy_hidden_dim
        )
    
    def forward(self,s):
        s = self.conv(s)
        for block in range(self.res_blocks_num):
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

def train(net, 
          train_datapath, 
          val_datapath=None, 
          use_mlflow=True,
          epochs=20, 
          seed=0, 
          save_path='./', 
          is_debug=False,
          adam_lr=3e-3,
          scheduler_gamma=0.2,
          batch_size=64,          
          ):
    print("Starting Training")
    torch.manual_seed(seed)
    cuda = torch.cuda.is_available()
    net.train()

    if use_mlflow:                
        mlflow.log_params({
            'model_name': net.name,
            "epochs": epochs,
            "adam_lr": adam_lr,
            "scheduler_gamma": scheduler_gamma,
            "batch_size": batch_size,            
        })

    criterion = AlphaLoss()
    optimizer = optim.Adam(net.parameters(), lr=adam_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 400], gamma=scheduler_gamma)

    pin_memory = cuda
    train_set = board_data(train_datapath)    
    if is_debug:
        train_subset = Subset(train_set, range(min(batch_size, len(train_set))))
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)    

    if val_datapath is not None:
        val_set = board_data(val_datapath)
        if is_debug:
            val_subset = Subset(val_set, range(min(64, len(val_set))))
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
        else:
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)

    losses_per_epoch = []

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0.0
        correct_policy = 0
        total_samples = 0
        batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for state, policy, value in progress_bar:
            if cuda:
                state, policy, value = state.cuda(), policy.cuda(), value.cuda()

            optimizer.zero_grad()
            policy_pred, value_pred = net(state)
            loss = criterion(value_pred[:, 0], value, policy_pred, policy)
            loss.backward()
            optimizer.step()

            # --- train metrics ---
            epoch_loss += loss.item()
            batches += 1
            pred_moves = policy_pred.argmax(dim=1)
            true_moves = policy.argmax(dim=1)
            correct_policy += (pred_moves == true_moves).sum().item()
            total_samples += state.size(0)

            progress_bar.set_postfix(loss=loss.item(), acc=correct_policy/total_samples if total_samples>0 else 0.0)

        if batches > 0:
            avg_loss = epoch_loss / batches
            train_acc = correct_policy / total_samples if total_samples > 0 else 0.0
            losses_per_epoch.append(avg_loss)
            tqdm.write(f"[Epoch {epoch+1}] Avg train loss = {avg_loss:.4f}, train policy acc = {train_acc:.4f}")
        else:
            tqdm.write("No batches in this epoch — skipping.")
            continue

        
        if val_datapath is not None:            
            net.eval()
            with torch.no_grad():
                val_loss = 0.0
                correct_policy_val = 0
                total_samples_val = 0
                val_batches = 0

                val_bar = tqdm(val_loader, desc="Validating", leave=False)
                for state, policy, value in val_bar:
                    if cuda:
                        state, policy, value = state.cuda(), policy.cuda(), value.cuda()
                    policy_pred, value_pred = net(state)
                    loss = criterion(value_pred[:, 0], value, policy_pred, policy)
                    val_loss += loss.item()
                    val_batches += 1

                    # Policy accuracy
                    pred_moves = policy_pred.argmax(dim=1)
                    true_moves = policy.argmax(dim=1)
                    correct_policy_val += (pred_moves == true_moves).sum().item()
                    total_samples_val += state.size(0)

                val_loss /= max(val_batches, 1)
                val_accuracy = correct_policy_val / total_samples_val if total_samples_val > 0 else 0.0
                tqdm.write(f"[Val] loss={val_loss:.4f}, policy_acc={val_accuracy:.4f}")

        scheduler.step()

        if use_mlflow:
            mlflow.log_metrics({                
                "train_loss": avg_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss if val_datapath else 0,
                "val_accuracy": val_accuracy if val_datapath else 0,
            }, step=epoch)

        # Early stopping
        if len(losses_per_epoch) > 5 and abs(losses_per_epoch[-1] - losses_per_epoch[-5]) < 1e-3:
            tqdm.write("Early stopping: loss plateau.")
            break

    # Plot losses
    plt.figure()
    plt.plot(range(1, len(losses_per_epoch) + 1), losses_per_epoch)
    plt.xlabel("Epoch")
    plt.ylabel("Avg Loss")
    plt.title("Training Loss per Epoch")
    if use_mlflow:
        mlflow.log_figure(plt.gcf(), f"Loss_vs_Epoch.png")        
    else:
        plt.savefig(os.path.join(save_path, "Loss_vs_Epoch.png"))
    print("Finished Training")




def test(net, test_datapath, batch_size=64, seed=0, is_debug=False):
    torch.manual_seed(seed)
    cuda = torch.cuda.is_available()
    net.eval()

    criterion = AlphaLoss()
    pin_memory = cuda
    test_set = board_data(test_datapath)
    if is_debug:
        test_subset = Subset(test_set, range(batch_size))
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
    else:
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
    

    total_loss = 0.0
    correct_policy = 0
    total_samples = 0
    batches = 0

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing", leave=False)
        for state, policy, value in progress_bar:
            if cuda:
                state, policy, value = state.cuda(), policy.cuda(), value.cuda()

            policy_pred, value_pred = net(state)
            loss = criterion(value_pred[:, 0], value, policy_pred, policy)
            total_loss += loss.item()
            batches += 1

            # Policy accuracy
            pred_moves = policy_pred.argmax(dim=1)
            true_moves = policy.argmax(dim=1)
            correct_policy += (pred_moves == true_moves).sum().item()
            total_samples += state.size(0)

            progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / max(batches, 1)
    accuracy = correct_policy / max(total_samples, 1)

    tqdm.write(f"[Test] loss = {avg_loss:.4f}, policy_acc = {accuracy:.4f}")    
    return avg_loss, accuracy




# def predict(net, board):
        