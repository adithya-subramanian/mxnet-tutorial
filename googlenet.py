import mxnet as mx
import numpy as np
import os
import logging

def to4d(img):
    return img.reshape(img.shape[0],3,224,224).astype(np.float32)/255

def Inception_naive(net,num_filter_1,num_filter_3,num_filter_5,stride_pool,type_of_pooling):
	conv_1 = mx.sym.Convolution(data = net,kernel = (1,1),num_filter = num_filter_1)
	conv_3 = mx.sym.Convolution(data = net,kernel = (3,3),num_filter = num_filter_3)
	conv_5 = mx.sym.Convolution(data = net,kernel = (5,5),num_filter = num_filter_5)
	pool_1 = mx.sym.Pooling(data = net,kernel = (3,3),pool_type = type_of_pooling,stride = stride_pool)
	return mx.sym.Concat([conv_1,conv_5,conv_3,pool_1])

def Inception(net,num_filter_1,num_filter_3_1,num_filter_5_1,num_filter_3,num_filter_5,stride_pool,type_of_pooling = "max"):
	conv_1_1 = mx.sym.Convolution(data = net,kernel = (1,1),num_filter = num_filter_3_1)
	conv_1_2 = mx.sym.Convolution(data = net,kernel = (1,1),num_filter = num_filter_5_1)
	pool_1 = mx.sym.Pooling(data = net,kernel = (3,3),pool_type = 'max',stride = stride_pool)
	conv = mx.sym.Convolution(data = net,kernel = (1,1),num_filter = num_filter_1)
	conv_2_1 = mx.sym.Convolution(data = pool_1,kernel = (1,1),num_filter = num_filter_1)
	conv_3 = mx.sym.Convolution(data = conv_1_1,kernel = (3,3),num_filter = num_filter_3)
	conv_5 = mx.sym.Convolution(data = conv_1_2,kernel = (5,5),num_filter = num_filter_5)
	return mx.sym.Concat([conv,conv_3,conv_5,conv_2_1],)

def Extra(net):
	net1 = mx.sym.Pooling(data = net,pool_type = "average",stride = (3,3),kernel = (5,5))
	net1 = mx.sym.Convolution(data = net,kernel = (1,1),stride = (1,1),num_filter = 128)
	net1 = mx.sym.Flatten(data = net1)
	net1 = mx.sym.FullyConnected(data = net1,num_hidden = 1024)
	net1 = mx.sym.Activation(data = net1,act_type = 'relu')
	net1 = mx.sym.FullyConnected(data = net1,num_hidden = len(dataset))
	return mx.sym.SoftmaxOutput(data = net1)

dataset = os.listdir('data/101')
train_images = []
train_label = []
j = 0 
for i in dataset:
	z = os.listdir('data/101'+'/'+i)
	for j in z:
		train_images.append(((mx.img.imdecode(open('data/101'+'/'+i+'/'+j).read())).asnumpy()).tolist())
		train_label.append(j)
	j = j + 1


batch_size = 100
train_iter = mx.io.NDArrayIter(to4d(train_img), train_lbl, batch_size, shuffle=True)
net = mx.sym.Variable('data')
net = mx.sym.Convolution(data = net,kernel = (7,7),stride = (2,2),num_filter = 64)
net = mx.sym.Pooling(data = net,pool_type = 'max',stride = (2,2),kernel = (3,3))
net = mx.sym.LRN(data = net)
net = mx.sym.Convolution(data = net,kernel = (1,1),stride = (1,1),num_filter = 64)
net = mx.sym.Convolution(data = net,kernel = (3,3),stride = (1,1),num_filter = 192)
net = mx.sym.LRN(data = net)
net = mx.sym.Pooling(data = net,pool_type = "max",stride = (2,2),kernel = (3,3))
net = Inception(net,64,96,16,128,32)
net = Inception(net,128,128,32,192,96)
net = mx.sym.Pooling(data = net,stride = (2,2),kernel = (3,3),pool_type = "max")
net = Inception(net,192,96,16,208,48)
softmax0 = Extra(net)
net = Inception(net,160,112,24,224,64)
net = Inception(net,128,128,24,256,64)
net = Inception(net,112,114,32,288,64)
softmax1 = Extra(net)
net = Inception(net,256,160,32,320,128)
net = mx.sym.Pooling(data = net,stride = (2,2),kernel = (3,3),pool_type = "max")
net = Inception(net,256,160,32,320,128)
net = Inception(net,384,192,48,384,128)
net = mx.sym.Pooling(data = net,stride = (1,1),kernel = (7,7),pool_type = "average")
net = mx.sym.FullyConnected(data = net,num_hidden = len(dataset))
net = mx.sym.SoftmaxOutput(data = net)

logging.getLogger().setLevel(logging.DEBUG)

model = mx.model.FeedForward(
    symbol = net,       # network structure
    num_epoch = 10,     # number of data passes for training 
    learning_rate = 0.1 # learning rate of SGD 
)
model.fit(
    X=train_iter,       # training data
    #eval_data=val_iter, # validation data
    batch_end_callback = mx.callback.Speedometer(batch_size, 200) # output progress for each 200 data batches
)
val_img = (mx.img.imdecode(open('location').read())).asnumpy()
prob = model.predict(val_img[0:1].astype(np.float32)/255)[0]
print prob.argmax(),max(prob)