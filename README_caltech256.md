# beeva-poc-mxnet
Proof of Concept with MXNet and GPUs

### Experiment: fine-tune with pre-trained models on multi-GPU

* Training (fine-tune) on mxnet single machine, multiple GPU
* Dataset: Caltech256. Approximately 30000 images, crop to 224x224. --num-classes 256 --num-examples 15240
* Models: ResNet-152

* Based on https://github.com/dmlc/mxnet/blob/master/docs/how_to/multi_devices.md#how-to-launch-a-job
* **Infrastructure 1**: AWS p2.8x (8 gpus nvidia Tesla K80). Deep Learning AMI Ubuntu Linux - 2.4_Oct2017 (ami-37bb714d), mxnet==0.11.0, NVIDIA Driver 375.66, CUDA 8.0, libcudnn.so.5.1.10


```
# cd src/mxnet/example/image-classification
# ./data/caltech256.sh
python fine-tune.py --pretrained-model imagenet11k-resnet-152 --gpus 0,1,2,3,4,5,6,7 --data-train caltech256-train.rec --data-val caltech256-val.rec --batch-size 128 --num-classes 256 --num-examples 15240 --num-epochs 6
```


#### Results:

| infrastructure | model | batch size | gpus | Accuracy (validation) | Epochs | Training time (s/epoch) | Throughput (samples/s) | GPU Utilization
| --- | --- | --- | --- | --- | --- | --- | --- | ---
| 1 | imagenet11k-resnet-152 | 16x8 = 128 | 8 | 0.865 | 6 | 102 | 150 | 83% (65% to 97%)
| 1 | imagenet11k-resnet-152 | 16x8 = 128 | 8 | 0.833 | 1 | 126 | 150 | 83% (65% to 97%)
| 1 | imagenet11k-resnet-152 | 16 (lr = 0.01) | 1 | 0.635 | 1 | 770 | 20 | 95%
| 1 | imagenet11k-resnet-152 | 16 (lr = 0.00125) | 1 | 0.836 | 1 | 770 | 20 | 95%
| --- | --- | --- | --- | --- | --- | --- | --- | ---
