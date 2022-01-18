# Edge-device-deployment
## [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/Asad-Ismail/Edge-device-deployment/issues) [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FAsad-Ismail%2FEdge-device-deployment&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

## General Pytorch Quantization Test
See quantization-pytorch for Pytorch Quantizatizationn Tests

## D2Go for Instance Segmentation
See D2Go for Instance semgementaiton on custom Vegetable dataset. 


**To Train on custom dataset run**
python veg_train_infer.py --train True

**To Validate and visualize the model**
python veg_train_infer.py --val True

**To Quantize the model**
python veg_train_infer.py --m_conversion True

Visual results on original dataset

<p align="center">
    <img src="images/org_model.png" alt="animated" width=650 height=500 />
  </p>
  
With Post taining quantization we observe the classification and object detection works well while we suffer significat drop in accuracy of Dense Prediction (masks)

<p align="center">
    <img src="images/post_quantization.png" alt="animated" width=650 height=500 />
  </p>

With Quantization Aware Pruning we see negligible drop in accuracy. The size of final quantized model is total **7.4MB**      

<p align="center">
    <img src="images/QAT_model.png" alt="animated" width=650 height=500 />
  </p>



## Android App for Instance Segmentation
For Android app for instance segmentaiton using MaskRCNN we added nearest Neighbout Interpolation for Masks and Canvas visualization of masks in Java. See D2Go/android-app for android app. On **Huwawei P30** model runs at average of **140ms**

<p align="center">
  <img src="images/android_sc_1.jpg" width="200" height=500 />
  <img src="images/android_sc_2.jpg" width="200" height=500 /> 
</p>

### References
```
1) See  D2G0 https://github.com/facebookresearch/d2go
2) See  Android App for Pytorch https://github.com/pytorch/android-demo-app
