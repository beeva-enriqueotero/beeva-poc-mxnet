# beeva-poc-mxnet
Proof of Concept with MXNet and GPUs

### Experiment: recommender system based on MxNet examples

Based on [MxNet recommender examples](https://github.com/apache/incubator-mxnet/tree/master/example/recommenders)

More references:
- [MXNet for Collaborative Deep Learning in Recommender Systems](https://github.com/dmlc/mxnet-notebooks/blob/master/python/recommendation_systems/cdl/collaborative-dl.ipynb)
- [Gluon Intro to Recommender Systems](http://gluon.mxnet.io/chapter11_recommender-systems/intro-recommender-systems.html)

Note: we have to convert rating prediction examples to learning to rank approach.

### Issues:
- MxNet output layers are not flexible to output probabilities. And this is required to apply crossentropy. [More info](https://github.com/apache/incubator-mxnet/issues/8807)
