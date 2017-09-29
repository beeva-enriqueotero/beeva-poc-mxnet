# beeva-poc-mxnet
Proof of Concept with MXNet and GPUs

### Experiment 1: multi-GPU

* Training on mxnet single machine, multiple GPU
* Dataset: CIFAR10. 60000 samples 32x32. Training set: 50000
* Models: ResNet-110, ResNet-50

* Based on https://github.com/dmlc/mxnet/blob/master/docs/how_to/multi_devices.md#how-to-launch-a-job
* **Infrastructure 1**: AWS p2.8x (8 gpus nvidia Tesla K80). Deep Learning 2.3 Ubuntu AMI, mxnet==0.11.0, NVIDIA Driver 375.66, CUDA 8.0, libcudnn.so.5.1.10
* **Infraestructure 2**: DGX1 (8 nvidia P100)

```
time python ./example/image-classification/train_cifar10.py --gpus 0 --num-epochs 12 --network resnet --batch-size 128 --disp-batches 20
time python ./example/image-classification/train_cifar10.py --gpus 0,1,2,3 --num-epochs 12 --network resnet --num-layers 50 --batch-size 256 --lr=0.2

```


#### Results:

| infrastructure | model | batch size | gpus | Accuracy (validation) | Epochs | Training time (s/epoch) | Throughput (samples/s)
| --- | --- | --- | --- | --- | --- | --- | ---
| 1 | resnet110 | 128 | 1 | 0.818 | 12 | 96.9 | 520
| 1 | resnet50 | 64x8 | 8 | 0.702 | 12 | 8.1s | 6200 
| 1 | resnet50 | 64x4 | 4 | 0.768 | 12 | 13.8s | 3600
| 1 | resnet50 | 64x2 | 2 | 0.817 | 12 | 27.1s | 1800
| 1 | resnet50 | 64x1 |   |  |  |  | 
| 1 | resnet50 | 64x8 (lr=0.4) | 8 | 0.765 | 12 | 8.0s | 6200
| 1 | resnet50 | 64x4 (lr=0.2) | 4 | 0.783 | 12 | 13.8s | 3600
| 1 | resnet50 | 512x1 (lr=0.05) | 1 | 0.703 | 12 | 36.3 | 1400
| 1 | resnet50 | 512x1 (lr=0.4) | 1 | 0.796 | 12 | 36.0 | 1400
| 1 | resnet50 | 512x8 (lr=3.2) | 8 | 0.278 | 12 | 4.5 | 11000
| 1 | resnet50 | 512 (lr=0.4) | 0 | ? | 12 | 110.0 | 450
| 2 | resnet50 | 512x8=4096 (lr=0.4) |8| 0.604 | 12 | 2.1s | 24000 (70%)
| 2 | resnet50 | 512x8=4096 (lr=0.8) |8| 0.599 | 12 | 2.1s | 24000 (70%)
| 2 | resnet110 | 512x8=4096 (lr=0.4) |8| 0.671 | 12 | 4.4s | 11000 (80%)


