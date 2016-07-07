# MLPModel.R

# define a multilayer perceptron
require(mxnet)
if(!exists("train")) { source("loadData.R") }


data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name = "fc1", num_hidden = 128)
act1 <- mx.symbol.Activation(fc1, name = "relu1", act_type = "relu")
fc2 <- mx.symbol.FullyConnected(act1, name = "fc2", num_hidden = 64)
act2 <- mx.symbol.Activation(fc2, name = "relu2", act_type = "relu")
fc3 <- mx.symbol.FullyConnected(act2, name = "fc3", num_hidden = 10)
softmax <- mx.symbol.SoftmaxOutput(fc3, name = "sm")
devices <- mx.cpu()   # using 1 CPU
mx.set.seed(42)

# Train the model with the training data
model <- mx.model.FeedForward.create(
  symbol = softmax,
  X = train.X,
  y = train.y,
  ctx = devices,
  num.round = 50,
  array.batch.size = 100,
  learning.rate = 0.07,
  momentum = 0.9,
  eval.metric = mx.metric.accuracy,
  initializer = mx.init.uniform(0.07),
  epoch.end.callback =  mx.callback.log.train.metric(100),
  array.layout = "rowmajor"
)