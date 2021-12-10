import torch
from torch.utils import data
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from utils import *
from models import resnet50model
from data_loader import fashion_mnist
import argparse
import os
from os import makedirs


parser = argparse.ArgumentParser()
parser.add_argument("--trydevice", type=str, help="Try device to train the netowrk can be cpu or cuda:x",default="cpu")
args=parser.parse_args()
out_dir="ressnet50_models"
makedirs(out_dir,exist_ok=True)


## Training Loop
def train(net,train_loader,test_loader):
    device=args.trydevice
    print(f"Trianing using Device {device}")
    lr=0.0001
    num_epochs=100
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    criteria = nn.CrossEntropyLoss()
    net.to(device)
    current_acc=0
    for epoch in tqdm(range(num_epochs),total=num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        net.train()
        train_loss = AverageMeter('Train_Loss', ':6.2f')
        for i, (X, y) in tqdm(enumerate(train_loader),total=len(train_loader)):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = criteria(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                train_loss.update(l)
        print(train_loss)
        test_acc = evaluate(net,criteria,test_loader,device=device)
        if test_acc.avg>current_acc:
            out_path=os.path.join(out_dir,f"best_weight_{test_acc.avg:.1f}.pth")
            torch.save(net.state_dict(), out_path)
            current_acc=test_acc.avg
        print(f"{test_acc}")

## Start Training
if __name__=="__main__":
    net=resnet50model()
    train_loader,test_loader=fashion_mnist()
    train(net,train_loader,test_loader)