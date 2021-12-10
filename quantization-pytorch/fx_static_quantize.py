import torch
from torch.quantization import get_default_qconfig, quantize_jit
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from models import resnet50model
import torch.nn as nn
from data_loader import fashion_mnist
from utils import *
import copy
from tqdm import tqdm
import argparse


# Setup warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.quantization'
)


parser = argparse.ArgumentParser()
parser.add_argument("--pretrain", type=str, help="Pretrain model to load")
parser.add_argument("--engine", type=str, default='qnnpack',help="Pretrain model to load")
args=parser.parse_args()


def calibrate(model, data_loader):
    print(f"Calibrating the model activations!!")
    model.eval()
    with torch.no_grad():
        for i,(image, target) in tqdm(enumerate(data_loader),total=len(data_loader)):
            model(image)
            if i>=100:
                break



if __name__=="__main__":
    ## Load trained model
    criterion = nn.CrossEntropyLoss()
    model=resnet50model()
    model_file=argparse.pretrain
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to("cpu")
    model.eval()
    ## Copy model
    model_to_quantize = copy.deepcopy(model)
    model_to_quantize.eval()
    qconfig = get_default_qconfig(args.engine)
    # set the qengine to control weight packing
    print(torch.backends.quantized.supported_engines)
    torch.backends.quantized.engine = args.engine
    qconfig_dict = {"": qconfig}
    # Preprare model fuse layers
    print(model_to_quantize)
    prepared_model = prepare_fx(model_to_quantize, qconfig_dict)
    print(prepared_model.graph)
    ## Load dataset for calibration
    train_loader,test_loader=fashion_mnist()
    calibrate(prepared_model, test_loader)  # run calibration on sample data
    # After calibration quantize the model
    quantized_model = convert_fx(prepared_model)
    print(quantized_model)
    
    print("Size of model before quantization")
    print_size_of_model(model)
    print("Size of model after quantization")
    print_size_of_model(quantized_model)
    
    top1= evaluate(model, criterion, test_loader,device="cpu")
    print("[Before serilaization Original Model] Evaluation accuracy on test dataset: %2.2f "%(top1.avg))
    
    top1= evaluate(quantized_model, criterion, test_loader,device="cpu")
    print("[Before serilaization Quantized Model] Evaluation accuracy on test dataset: %2.2f "%(top1.avg))
    
    ## Save serialized traced model
    fx_graph_mode_model_file_path = "resnet50_fx_graph_mode_quantized.pth"
    
    torch.jit.save(torch.jit.script(quantized_model), fx_graph_mode_model_file_path)
    loaded_quantized_model = torch.jit.load(fx_graph_mode_model_file_path)
    top1 = evaluate(loaded_quantized_model, criterion, test_loader,device="cpu")
    print("[After serialization/deserialization] Evaluation accuracy on test dataset: %2.2f"%(top1.avg))