# beeva-poc-mxnet
Proof of Concept with MXNet and GPUs

### Experiment: fine-tune with pre-trained models on multi-GPU

* Training (fine-tune) on mxnet single machine, multiple GPU
* Dataset: Caltech256. Approximately 30000 images, crop to 224x224. --num-classes 256 --num-examples 15240
* Models: ResNet-152

* Based on https://github.com/apache/incubator-mxnet/tree/master/example/image-classification#fine-tune-another-dataset and https://github.com/dmlc/mxnet-notebooks/blob/master/python/how_to/finetune.ipynb

* **Infrastructure 1**: AWS p2.8x (8 gpus nvidia Tesla K80). Deep Learning AMI Ubuntu Linux - 2.4_Oct2017 (ami-37bb714d), mxnet==0.11.0, NVIDIA Driver 375.66, CUDA 8.0, libcudnn.so.5.1.10

* **Infrastructure 2a**: AWS p3.8x (4 gpu nvidia V100). NVIDIA Volta Deep Learning AMI-46a68101-e56b-41cd-8e32-631ac6e5d02b-ami-655e831f.4 (ami-4cc11e36), nvcr.io mxnet:17.10, NVIDIA Driver 384.81, CUDA 9.0, no libcudnn

* **Infrastructure 2b**: AWS p3.8x (4 gpu nvidia V100). (community) NVIDIA Volta Deep Learning AMI-46a68101-e56b-41cd-8e32-631ac6e5d02b-ami-655e831f.4 (ami-4cc11e36), mxnet-cu90==1.0.0, NVIDIA Driver 384.90, CUDA 9.0, no libcudnn

* **Infrastructure 2c**: AWS p3.8x (4 gpu nvidia V100). (community) Deep Learning Base AMI (Ubuntu) Version 2.0 (ami-10ef8d6a), mxnet-cu90==0.12.0, NVIDIA Driver 384.81, CUDA 9.0, no libcudnn

* **Infrastructure 2d**: AWS p3.8x (4 gpu nvidia V100). (community) Deep Learning Base AMI (Ubuntu) Version 2.0 (ami-10ef8d6a), [mxnet 1.0.0](https://github.com/apache/incubator-mxnet/releases/tag/1.0.0), build with USE_NCCL=0, kvstore=’device’, NVIDIA Driver 384.81, CUDA 9.0, no libcudnn

* **Infrastructure 2e**: AWS p3.8x (4 gpu nvidia V100). (community) Deep Learning Base AMI (Ubuntu) Version 2.0 (ami-10ef8d6a), [mxnet 1.0.0](https://github.com/apache/incubator-mxnet/releases/tag/1.0.0), build with USE_NCCL=1, kvstore=’nccl’, NVIDIA Driver 384.81, CUDA 9.0, no libcudnn



#### Results:

| infrastructure | model | batch size & lr | gpus | Accuracy (validation) | Epochs | Training time (s/epoch) | Throughput (samples/s) | GPU Utilization
| --- | --- | --- | --- | --- | --- | --- | --- | ---
| 1 | imagenet11k-resnet-152 | 16 (0.00125) | 1 | 0.839 | 1 |  | 18.9 | 
| 1 | imagenet11k-resnet-152 | 16 (0.00125) | 1 | 0.842 | 1 |  | 18.9 | 



