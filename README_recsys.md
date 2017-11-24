# beeva-poc-mxnet
Proof of Concept with MXNet and GPUs

### Experiment: recommender system based on MxNet examples

Note: we have to convert rating prediction examples to learning to rank approach.

### Issues:
- MxNet output layers are not flexible to output probabilities. And this is required to apply crossentropy. [More info](https://github.com/apache/incubator-mxnet/issues/8807)
