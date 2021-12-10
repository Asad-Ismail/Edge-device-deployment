import torchvision
import torch
import torchvision.transforms as T


def fashion_mnist():
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    gray2rgb=T.Lambda(lambda x: x.float().repeat(3, 1, 1))    

    fashion_data_train = torchvision.datasets.FashionMNIST('.',download=True,train=True,transform=T.Compose([T.Resize(size=(224,224)),T.PILToTensor(),gray2rgb,normalize]))
    train_loader = torch.utils.data.DataLoader(fashion_data_train,
                                            batch_size=32,
                                            shuffle=True)

    fashion_data_validate = torchvision.datasets.FashionMNIST('.',download=True,train=False,transform=T.Compose([T.Resize(size=(224,224)),T.PILToTensor(),gray2rgb,normalize]))
    test_loader = torch.utils.data.DataLoader(fashion_data_validate,
                                            batch_size=32,
                                            shuffle=True)

    return train_loader, test_loader