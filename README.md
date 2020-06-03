# PPSGAN
Selective Feature Anonymization for Privacy-Preserving Image Data Publishing

There is a strong positive correlation between the development of deep learning and the amount of public data available. Not all data can be released in their raw form because of the risk to the privacy of the related individuals. The main objective of privacy-preserving data publication is to anonymize the data while maintaining their utility. In this paper, we propose a privacy-preserving semi-generative adversarial network (PPSGAN) that selectively adds noise to class-independent features of each image to enable the processed image to maintain its original class label. Our experiments on training classifiers with synthetic datasets anonymized with various methods confirm that PPSGAN shows better utility than other conventional methods, including blurring, noise-adding, filtering, and generation using GANs.

T. Kim and J. Yang, "Selective Feature Anonymization for Privacy-Preserving Image Data Publishing," in Electronics.

URL: https://www.mdpi.com/2079-9292/9/5/874

![alt text](https://github.com/tgisaturday/PPSGAN/blob/master/figure1.png)

## Training

```
train_[...].py [dataset_name] [model_name] [previous_iteration(0 if initial training)]
```
