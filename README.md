# beeva-poc-mxnet
Proof of Concept with MXNet and GPUs

### Experiment 1: multi-GPU

* Training on mxnet single machine, multiple GPU
* Dataset: MNIST. 60000 train samples, 10000 test samples
* Model: LeNet
* Based on https://github.com/dmlc/mxnet/blob/master/docs/how_to/multi_devices.md#how-to-launch-a-job
* **Infrastructure 1**: AWS p2.8x (8 gpus nvidia Tesla K80). Deep Learning 2.0 Amazon Community AMI, mxnet==0.9.3, CUDA 7.5, libcudnn.so.5
* **Infrastructure 2**: Google n1-standard-16 with 2 gpus (nvidia Tesla K80), [mxnet](https://github.com/dmlc/mxnet/commit/01b808b88b9f3f3a998541c538ec388d660e4a7c), NVIDIA Driver 375.39, CUDA 8.0, libcudnn.so.5 (CuDNN 5.1). [How to configure drivers](https://github.com/beeva-enriqueotero/beeva-poc-distributed-tensorflow/blob/master/README_multigpu.md#deploy)
* **Infrastructure 3**: Google n1-highmem-32 with 8 gpus (nvidia Tesla K80), mxnet_cu80-0.10.0-py2.py3-none-manylinux1_x86_64.whl, [mxnet](https://github.com/dmlc/mxnet/commit/3ceb6d2f91121d5ffa5b81f435e8bcfcc1a75792), NVIDIA Driver 375.66, CUDA 8.0, libcudnn.so.5 (CuDNN 5.1). [How to configure drivers](https://github.com/beeva-enriqueotero/beeva-poc-distributed-tensorflow/blob/master/README_multigpu.md#deploy)
* **Infrastructure 4**: Azure NC6 with 1 gpu (nvidia Tesla K80), mxnet==0.9.5 (mxnet-0.9.5-py3.5), [mxnet](https://github.com/apache/incubator-mxnet/commit/0768c0e97c5aa1d142ff0b3b8d37b1c736a42b83), Release 0.10.0.post2, NVIDIA Driver v367.48, CUDA 8.0 (V8.0.61), libcudnn.so.5.1.10 (CuDNN 5.1)
* **Infrastructure 5**: Azure NC24 with 4 gpu (nvidia Tesla K80), mxnet==0.9.5 (mxnet-0.9.5-py3.5), [mxnet](https://github.com/apache/incubator-mxnet/commit/0768c0e97c5aa1d142ff0b3b8d37b1c736a42b83), Release 0.10.0.post2, NVIDIA Driver v367.48, CUDA 8.0 (V8.0.61), libcudnn.so.5.1.10 (CuDNN 5.1)

```
time python ./example/image-classification/train_mnist.py --gpus 0,1,2 --num-epochs 12 --network lenet --batch-size 512 --disp-batches 50
```


#### Results:

| infrastructure | model | batch size | gpus | Accuracy (validation) | Epochs | Training time (s/epoch)
| --- | --- | --- | --- | --- | --- | ---
| 1 | lenet | 64 | 1 | 0.991 | 12 | 3.1 (43.7-6)/12 (19200 samples/s) 
| 1 | lenet | 64 | 2 | 0.992 | 12 | 2.4 (35.5-6)/12 (25500 samples/s)
| 1 | lenet | 64 | 3 | 0.992 | 12 | 2.3 (35.2-6)/12 (26500 samples/s)
| 1 | lenet | 64 | 4 | 0.991 | 12 | 2.9 (44.4-6)/12 (20500 samples/s)
| 1 | lenet | 64 | 8 | 0.992 | 12 | 5.7 (85.0-6)/12 (10500 samples/s)
| 1 | lenet | 128 | 1 | 0.991 | 12 | 2.4 (34.1-6)/12 (25000 samples/s) 
| 1 | lenet | 128 | 2 | 0.992 | 12 | 1.6 (25.2-6)/12 (37500 samples/s)
| 1 | lenet | 128 | 3 | 0.992 | 12 | 1.4 (23.6-6)/12 (43000 samples/s)
| 1 | lenet | 128 | 4 | 0.992 | 12 | 1.5 (24.6-6)/12 (41000 samples/s)
| 1 | lenet | 128 | 8 | 0.992 | 12 | 2.9 (46.5-6)/12 (21000 samples/s)
| 1 | lenet | 128 | 0 | 0.992 | 12 | 40.0 (578.0-6)/12 (1500 samples/s)
| 1 | mlp | 128 | 1 | 0.979 | 12 | 0.8 (14.1-6)/12 (80000 samples/s)
| 1 | mlp | 128 | 2 | 0.983 | 12 | 0.8 (14.1-6)/12 (80500 samples/s)
| 1 | mlp | 128 | 3 | 0.982 | 12 | 1.0 (18.3-6)/12 (59000 samples/s)
| 1 | mlp | 128 | 4 | 0.981 | 12 | 1.3 (22.3-6)/12 (46000 samples/s)
| 1 | mlp | 128 | 8 | 0.981 | 12 | 2.3 (39.3-6)/12 (25500 samples/s)
| 2 | lenet | 128 | 1 | 0.992 | 12 | 2.7 (36.4-6)/12 (22500 samples/s) 
| 2 | lenet | 128 | 2 | 0.992 | 12 | 2.0 (30.3-6)/12 (29500 samples/s)
| 2 | lenet | 128 | 0 | 0.992 | 12 | 120.0 (1500?-6)/12 (500 samples/s)
| 3 | lenet | 128 | 0 | 0.992 | 12 | 59.5 (1200?-6)/12 (1000 samples/s)
| 3 | lenet | 128 | 1 | 0.992 | 12 | 2.5 (38.8-6)/12 (24500 samples/s)
| 3 | lenet | 128 | 2 | 0.992 | 12 | 2.4 (35.7-6)/12 (28800 samples/s)
| 3 | lenet | 128 | 4 | 0.992 | 12 | 2.4 (43.5-6)/12 (25000 samples/s)
| 3 | lenet | 128 | 8 | 0.991 | 12 | 2.4 (60.0-6)/12 (19500 samples/s)
| 3 | lenet | 256 | 8 | 0.992 | 12 | 1.4 (39.1-6)/12 (42500 samples/s)
| 3 | lenet | 512 | 1 | 0.990 | 12 | 1.7 (28.2-6)/12 (35000 samples/s)
| 3 | lenet | 512 | 8 | 0.990 | 12 | 0.8 (29.6-6)/12 (76000 samples/s)
| 3 | lenet | 1024 | 1 | 0.963 | 12 | (9.2-6)/12 (? samples/s)
| 3 | lenet | 1024 | 8 | 0.990 | 12 | 0.8 (29.6-6)/12 (76500 samples/s)
| 3 | lenet | 2048 | 8 | 0.987 | 12 | (24.0-6)/12 (? samples/s)
| 1 | lenet | 512 | 0 | 0.991 | 12 | 33.0 (1800 samples/s)
| 1 | lenet | 512 | 1 | 0.991 | 12 | 1.7 (36000 samples/s)
| 1 | lenet | 512 | 8 | 0.991 | 12 | 0.6 (104000 samples/s)
| 4 | lenet | 256 | 1 | 0.992 | 12 | 2.1 (30000 samples/s)
| 4 | lenet | 512 | 0 | error | 12 | error
| 4 | lenet | 512 | 1 | 0.991 | 12 | 1.9 (32000 samples/s)
| 4 | lenet | 1024 | 1 | 0.990 | 12 | 1.8 (34000 samples/s)
| 4 | lenet | 2048 | 1 | 0.987 | 12 | 1.8 (34000 samples/s)
| 5 | lenet | 2048 | 4 | 0.988 | 12 | 0.6 (105000 samples/s)
| 5 | lenet | 2048 (lr=0.07) | 4 | 0.988 | 12 | 0.6 (105000 samples/s)
| 5 | lenet | 2048 (lr=0.07, lr-step-epochs=13) | 4 | 0.988 | 12 | 0.6 (105000 samples/s)


#### Conclusions:
* 1 NVIDIA Tesla K80 is 20x faster than Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

#### To Do:
* Test new AWS [g3 instances](https://blog.dominodatalab.com/new-g3-instances-in-aws-worth-it-for-ml/) 
* Check MxNet works with last update in (Community) Deep Learning AMI Ubuntu Linux - 1.4_Jun2017 (CuDNN 6.0)

#### Speed issues:
* See https://github.com/dmlc/mxnet/blob/master/docs/how_to/faq.md#speed
