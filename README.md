# beeva-poc-mxnet
Proof of Concept with MXNet and GPUs

### Experiment 1: multi-GPU

* Training on mxnet single machine, multiple GPU
* Dataset: MNIST. 60000 train samples, 10000 test samples
* Model: LeNet
* Based on https://github.com/dmlc/mxnet/blob/master/docs/how_to/multi_devices.md#how-to-launch-a-job
* **Infrastructure 1**: AWS p2.8x (8 gpus nvidia Tesla K80). Deep Learning 2.0 AMI, mxnet==0.9.3, libcudnn.so.5

```
time python train_mnist.py --gpus 0,1,2 --num-epochs 12 --network lenet --batch-size 128
```


#### Results:

| infrastructure | batch size | gpus | Accuracy (validation) | Epochs | Training time (s/epoch)
| --- | --- | --- | --- | --- | ---
| 1 | 64 | 1 | 0.991 | 12 | 3.1=(43.7-6)/12 (19200 samples/s) 
| 1 | 64 | 2 | 0.992 | 12 | 2.4=(35.5-6)/12 (25500 samples/s)
| 1 | 64 | 3 | 0.992 | 12 | 2.3=(35.2-6)/12 (26500 samples/s)
| 1 | 64 | 4 | 0.991 | 12 | 2.9=(44.4-6)/12 (20500 samples/s)
| 1 | 64 | 8 | 0.992 | 12 | 5.7=(85.0-6)/12 (10500 samples/s)
| 1 | 64 | 1 | 0.991 | 12 | 2.4=(34.1-6)/12 (25000 samples/s) 
| 1 | 64 | 2 | 0.992 | 12 | 1.6=(25.2-6)/12 (37500 samples/s)
| 1 | 64 | 3 | 0.992 | 12 | 1.4=(23.6-6)/12 (43000 samples/s)
| 1 | 64 | 4 | 0.992 | 12 | 1.5=(24.6-6)/12 (41000 samples/s)
| 1 | 64 | 8 | 0.992 | 12 | 2.9=(46.5-6)/12 (21000 samples/s)


#### Conclusions: 
* With AWS p2.8x 3 gpus is the fastest configuration. Between 25% and 40% faster than 1 gpu
* With AWS 4 gpus is slightly faster than 1. And 8 gpus is the slowest

