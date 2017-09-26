# beeva-poc-mxnet
Proof of Concept with MXNet and GPUs

### Experiment 1: multi-GPU

* Training on mxnet single machine, multiple GPU
* Dataset: CIFAR10. 60000 samples 32x32. Training set: 50000
* Models: ResNet-110, ResNet-50

* Based on https://github.com/dmlc/mxnet/blob/master/docs/how_to/multi_devices.md#how-to-launch-a-job
* **Infrastructure 1**: AWS p2.8x (8 gpus nvidia Tesla K80). Deep Learning 2.3 Ubuntu AMI, mxnet==0.11.0, NVIDIA Driver 375.66, CUDA 8.0, libcudnn.so.5.1.10

```
time python ./example/image-classification/train_cifar10.py --gpus 0 --num-epochs 12 --network resnet --batch-size 128 --disp-batches 20

```


#### Results:

| infrastructure | model | batch size | gpus | Accuracy (validation) | Epochs | Training time (s/epoch)
| --- | --- | --- | --- | --- | --- | ---
| 1 | resnet110 | 128 | 1 | 0.818 | 12 | 96.9 520 samples/s) 
| 1 | resnet50 | 64x8 | 8 | 0.702 | 12 | 8.1s 6200 samples/s) 

