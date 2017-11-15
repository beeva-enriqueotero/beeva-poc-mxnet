# beeva-poc-mxnet
Proof of Concept with MXNet and GPUs

### Experiment: fine-tune with pre-trained models on multi-GPU

* Training (fine-tune) on mxnet single machine, multiple GPU
* Dataset: Caltech256. Approximately 30000 images, crop to 224x224. --num-classes 256 --num-examples 15240
* Models: ResNet-152

* Based on https://github.com/apache/incubator-mxnet/tree/master/example/image-classification#fine-tune-another-dataset and https://github.com/dmlc/mxnet-notebooks/blob/master/python/how_to/finetune.ipynb

* **Infrastructure 1**: AWS p2.8x (8 gpus nvidia Tesla K80). Deep Learning AMI Ubuntu Linux - 2.4_Oct2017 (ami-37bb714d), mxnet==0.11.0, NVIDIA Driver 375.66, CUDA 8.0, libcudnn.so.5.1.10


* **Infrastructure 1b**: AWS p2.8x (8 gpus nvidia Tesla K80). Deep Learning AMI Ubuntu Linux - 2.4_Oct2017 (ami-37bb714d), mxnet==0.11.0, NVIDIA Driver 375.66, CUDA 8.0, libcudnn.so.5.1.10. [Boosted](http://docs.aws.amazon.com/AWSEC2/latest/WindowsGuide/optimize_gpu.html)

* **Infrastructure 2**: AWS p3.8x (4 gpus nvidia Volta V100). Deep Learning AMI Ubuntu Linux - 2.4_Oct2017 (ami-37bb714d), mxnet==0.11.0, NVIDIA Driver 375.66, CUDA 8.0, libcudnn.so.5.1.10

* **Infrastructure 3**: AWS p3.2x (1 gpu nvidia V100). NVIDIA Volta Deep Learning AMI-46a68101-e56b-41cd-8e32-631ac6e5d02b-ami-655e831f.4 (ami-4cc11e36), nvcr.io mxnet:17.10, NVIDIA Driver 384.81, CUDA 9.0, no libcudnn

* **Infrastructure 3b**: AWS p3.2x (1 gpu nvidia V100). (community) NVIDIA Volta Deep Learning AMI-46a68101-e56b-41cd-8e32-631ac6e5d02b-ami-655e831f.4 (ami-4cc11e36), mxnet-cu90==0.12.0, NVIDIA Driver 384.90, CUDA 9.0, no libcudnn

* **Infrastructure 4**: AWS p3.8x (4 gpu nvidia V100). NVIDIA Volta Deep Learning AMI-46a68101-e56b-41cd-8e32-631ac6e5d02b-ami-655e831f.4 (ami-4cc11e36), nvcr.io mxnet:17.10, NVIDIA Driver 384.81, CUDA 9.0, no libcudnn
```
# If infrastructure 3 or 4 (needs 2 to 10 mins to pull the container)
# nvidia-docker run -it nvcr.io/nvidia/mxnet:17.10 /bin/bash
# cd /opt/mxnet/example/image-classification
# pip install future
# ./data/caltech256.sh
```

```
# If infrastructure 3b (needs 5 mins to install dependencies and reboot)
# sudo apt install libgfortran3 libsm6 libxrender1 cuda9 nvidia-384 python-dev virtualenv
# virtualenv mxnet
# source mxnet/bin/activate
# pip install mxnet-cu90 opencv-python future
# git clone https://github.com/apache/incubator-mxnet
# sudo reboot
# ce incubator-mxnet/example/image-classification
# ./data/caltech256.sh
```

```
# cd src/mxnet/example/image-classification
# ./data/caltech256.sh
python fine-tune.py --pretrained-model imagenet11k-resnet-152 --gpus 0,1,2,3,4,5,6,7 --data-train caltech256-train.rec --data-val caltech256-val.rec --batch-size 256 --num-classes 256 --num-examples 15240 --num-epochs 1 --lr 0.02
python fine-tune.py --pretrained-model imagenet11k-resnet-152 --gpus 0 --data-train caltech256-train.rec --data-val caltech256-val.rec --batch-size 16 --num-classes 256 --num-examples 15240 --num-epochs 1 --lr 0.00125
python fine-tune.py --pretrained-model imagenet1k-resnet-50 --gpus 0,1,2,3,4,5,6,7 --data-train caltech256-train.rec --data-val caltech256-val.rec --batch-size 128 --num-classes 256 --num-examples 15240 --num-epochs 6 --lr 0.01
```


#### Results:

| infrastructure | model | batch size | gpus | Accuracy (validation) | Epochs | Training time (s/epoch) | Throughput (samples/s) | GPU Utilization
| --- | --- | --- | --- | --- | --- | --- | --- | ---
| 1 | imagenet11k-resnet-152 | 16x8 = 128 | 8 | 0.865 | 6 | 102 | 150 | 83% (65% to 97%)
| 1 | imagenet11k-resnet-152 | 16x8 = 128 | 8 | 0.833 | 1 | 126 | 150 | 83% (65% to 97%)
| 1 | imagenet11k-resnet-152 | 16 (lr = 0.01) | 1 | 0.635 | 1 | 770 | 20 | 95%
| 1 | imagenet11k-resnet-152 | 16 (lr = 0.00125) | 1 | 0.836 | 1 | 770 | 20 | 95%
| 1 | imagenet11k-resnet-152 | 16 (lr = 0.00125) | 0 |  |  |  | 5 | 0% (1700% cpu)
| --- | --- | --- | --- | --- | --- | --- | --- | ---
| 1 | imagenet11k-resnet-152 | 32 (lr = 0.0025) | 0 |  |  |  | 6 | 0% (1700% cpu)
| 1 | imagenet11k-resnet-152 | 32 (lr = 0.0025) | 1 |  |  |  | 21 | 97%
| 1 | imagenet11k-resnet-152 | 32x8 = 256 (lr = 0.02) | 8 | 0.825+-0.05 | 1 | 121 | 160 | 97%
| 1b | imagenet11k-resnet-152 | 32x8 = 256 (lr = 0.02) | 8 | 0.810 | 1 | 119.5+-0.5 | 160 | 97%
| --- | --- | --- | --- | --- | --- | --- | --- | ---
| --- | --- | --- | --- | --- | --- | --- | --- | ---
| 1 | imagenet11k-resnet-152 | 64 | 1 | outofmemory |  |  | 21 | 97%
| --- | --- | --- | --- | --- | --- | --- | --- | ---
| 1 | imagenet1k-resnet-50 | 16x8 = 128 | 8 | 0.755 | 6 | 43 | 360 | 89% (85% to 97%)
| 1 | imagenet1k-resnet-50 | 16 | 1 |  |  |  | 48 | 96%
| 1 | imagenet1k-resnet-50 | 16 | 0 |  |  |  | 11 | 0% (1800% cpu)
| --- | --- | --- | --- | --- | --- | --- | --- | ---
| 2 | imagenet11k-resnet-152 | 16x8 = 128 | 1 to 4 | cudaErrorCudartUnloading CUDA: unknown error
| --- | --- | --- | --- | --- | --- | --- | --- | ---
| 3 | imagenet11k-resnet-152 | 16 (lr = 0.00125) | 1 | 0.835+-0.006 | 1 | 147+-1 | 104 | 82%
| 3 | imagenet11k-resnet-152 | 32 (lr = 0.0025) | 1 | 0.838+-0.004 | 1 | 118 | 131 (median 130.78) | 89%
| 3 | imagenet11k-resnet-152 | 64 (lr = 0.005) | 1 | outofmemory |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | ---
| 3 | imagenet1k-resnet-50 | 16 (lr = 0.00125)| 1 | 0.271 | 1 | 60 | 260 | 86%
| 3 | imagenet1k-resnet-50 | 32 (lr = 0.0025)| 1 | 0.338 | 1 | 51 | 300 | 90%
| 3 | imagenet1k-resnet-50 | 64 (lr = 0.005)| 1 | 0.379 | 1 | 46 | 335 | 95%
| 3 | imagenet1k-resnet-50 | 128 (lr = 0.01)| 1 | outofmemory | |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | ---
| 3b | imagenet11k-resnet-152 | 16 (lr = 0.00125) | 1 | 0.826 | 1 | 149 | 104 | 82%
| 3b | imagenet11k-resnet-152 | 32 (lr = 0.0025) | 1 | 0.824+-0.006 | 1 | 118 | 131 (median 130.79) | 89%
| 3b | imagenet11k-resnet-152 | 64 (lr = 0.005) | 1 | outofmemory |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | ---
| 3b | imagenet1k-resnet-50 | 16 (lr = 0.00125)| 1 | 0.242+-0.001 | 1 | 60 | 260 | 86%
| 3b | imagenet1k-resnet-50 | 32 (lr = 0.0025)| 1 | 0.304+-0.001 | 1 | 51 | 300 | 90%
| 3b | imagenet1k-resnet-50 | 64 (lr = 0.005)| 1 | 0.333+-0.001 | 1 | 46 | 335 | 95%
| 3b | imagenet1k-resnet-50 | 128 (lr = 0.01)| 1 | outofmemory | |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | ---
| 4 | imagenet11k-resnet-152 | 16 (lr = 0.00125) | 1 | 0.835 | 1 | 147 | 104 | 82%
| 4 | imagenet11k-resnet-152 | 32 (lr = 0.0025) | 1 | 0.843 | 1 | 119 | 130 | 89%
| 4 | imagenet11k-resnet-152 | 4x16 (lr = 0.005) | 4 | 0.835+-0.001 | 1 | 41 | 410 | 4x 80%
| 4 | imagenet11k-resnet-152 | 4x16 (lr = 0.005 dtype = float16) | 4 | 0.830+-0.004 | 1 | 41 | 410 | 4x 80%
| 4 | imagenet11k-resnet-152 | 4x32 (lr = 0.01) | 4 | 0.836+-0.001 | 1 | 34 | 520 | 4x 86%
| 4 | imagenet11k-resnet-152 | 4x32 (lr = 0.01 dtype = float16) | 4 | 0.818+-0.007 | 1 | 34 | 520 | 4x 88%
| 4 | imagenet11k-resnet-152 | 256 (lr = 0.02)| 4 | outofmemory | |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | ---
| 4 | imagenet1k-resnet-50 | 16 (lr = 0.00125)| 1 | 0.271 | 1 | 60 | 260 | 86%
| 4 | imagenet1k-resnet-50 | 32 (lr = 0.0025)| 1 | 0.337 | 1 | 51 | 300 | 90%
| 4 | imagenet1k-resnet-50 | 64 (lr = 0.005)| 1 | 0.363 | 1 | 46 | 335 | 95%
| 4 | imagenet1k-resnet-50 | 4x16 = 64 (lr = 0.005)| 4 | 0.275 | 1 | 16 | 1010 | 81%
| 4 | imagenet1k-resnet-50 | 4x32 = 128 (lr = 0.01)| 4 | 0.341 | 1 | 14 | 1200 | 88%
| 4 | imagenet1k-resnet-50 | 4x64 = 256 (lr = 0.02)| 4 | 0.372 | 1 | 13 | 1340 | 94%
| 4 | imagenet1k-resnet-50 | 4x128 (lr = 0.04)| 4 | outofmemory | |  |  |


#### Conclusions:
- P3 instances have great performance and cost effective on-demand prices :)
- Still their scarce availability affects spot market :(
- We got similar throughput without Nvidia MxNet optimized container, but container provides between 2% to 10% more accuracy
- We were unable to get benefits with float16. More [info](https://github.com/apache/incubator-mxnet/issues/7778)


