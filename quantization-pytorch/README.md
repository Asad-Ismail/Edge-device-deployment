## Pytorch Quantization tests

### Tested Graph mode quantization of pytorch which takes care of most of the intricaicies like merging layers during static quanitazation
## Requirement:

Quantization decreases the model size by 4x and also require less memory bandwidth leading to considerable speedups

## Results:
Renet50 model is trained on fashion mnist for 100 epochs. Original trained FP 32 model achived test accuracy of 93.4%.
We will consider three different types of quantization:

1) Dynamic quantization
2) Static quantization
3) Quantization aware training (QAT)

## Dynamic Quantization:
Dynamic Quantization is not suitable for CNNs becuase dynamic quantization quantizes and calcualte the quantized activation ranges on the fly, since the CNNs activations are memory bound this operation is quite expensive.
## Static Quantization:
Static Quantization requires a calibration dataset to caculate the ranges of activations. For mnist after calibration it  has test accuracy of 85.3 with engine qnnpack and 88.86 for FBgemm backend
## Quantization aware Training:
Quntization aware training can retain most of the accuracy dropeed during static qntization. Quantization aware trained model has test accruacy of 93.4%. We see almost no loss in accurcy with quantization aware training.

The two engines which can be used for quantization are fbgemm and qnnx. FBgemm can be used for x86 processors wchich supports AVX instruction set. On the other hand qnnx engine are used for model deployment in arm processors which most of the edge devices have. The QAT model was traind using both engines for FBGEMM the test accuracy was 93.4% where as for Qnnpack engine the test accuracy is 93.0  ~.4% loss in accuracy of original FP32 models. The accuracy drop is small enough not sure if it is because of the bakend engine or because of random initilizations and stochastic nature of stocashtic gradient descent. It needs more research.

