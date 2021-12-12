from os import makedirs
import torch
from torch.quantization import get_default_qconfig, quantize_jit
from torch.quantization.quantize_fx import prepare_fx, convert_fx
import torch.quantization.quantize_fx as quantize_fx
from models import resnet50model
from data_loader import fashion_mnist
from tqdm import tqdm
import torch.nn as nn
from utils import *
import copy
import argparse
import os
# Use Quantization Aware Training


parser = argparse.ArgumentParser()
parser.add_argument("--pretrain", type=str, help="Pretrain model to load")
parser.add_argument("--engine", type=str, default='qnnpack',help="Pretrain model to load")
parser.add_argument("--trydevice", type=str, help="Try device to train the netowrk can be cpu or cuda:x",default="cpu")
args=parser.parse_args()

out_dir="qat_models_qnn"
makedirs(out_dir,exist_ok=True)



## Training Loop
def train(net,train_loader,test_loader):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device= args.trydevice
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
        model_to_quantize = copy.deepcopy(net)
        test_acc = evaluate(model_to_quantize,criteria,test_loader,device=device)
        if test_acc.avg>current_acc:
            print("[Before Quantization ] Evaluation accuracy on test dataset: %2.2f"%(test_acc.avg))
            model_quantized = quantize_fx.convert_fx(model_to_quantize)
            top1 = evaluate(model_quantized, criteria, test_loader,device="cpu")
            print("[After Quantization ] Evaluation accuracy on test dataset: %2.2f"%(top1.avg))
            out_path=os.path.join(out_dir,f"qat_best_weight_{top1.avg:.1f}.pth")
            torch.jit.save(torch.jit.script(model_quantized),  out_path)
            current_acc=test_acc.avg
        

## Start Training
if __name__=="__main__":
    #model
    model=resnet50model()
    model_file=args.pretrain
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)

    model_to_quantize = copy.deepcopy(model)
    qconfig_dict = {"": torch.quantization.get_default_qat_qconfig(args.engine)}
    model_to_quantize.train()
    # prepare
    net= quantize_fx.prepare_qat_fx(model_to_quantize, qconfig_dict)
    torch.backends.quantized.engine = args.engine

    #data
    train_loader,test_loader=fashion_mnist()
    train(net,train_loader,test_loader)
