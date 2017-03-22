import mxnet as mx
import numpy as np
import os
import logging

def to4d(img):
    return img.reshape(img.shape[0],3,224,224).astype(np.float32)/255
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

net =  mx.sym.Variable('data')
net1 = mx.sym.Convolution(data = net,kernel = (7,7),stride = (2,2),num_filter = 64)
net2 = mx.sym.Activation(data = mx.sym.BatchNorm(net1),act_type = "relu")
net3 = mx.sym.Pooling(data = net2,pool_type = "max",stride = (2,2),kernel = (2,2))
net4 = mx.sym.Convolution(data = net3,kernel = (3,3),stride = (2,2),num_filter = 64)
net5 = mx.sym.Activation(data = mx.sym.BatchNorm(net4),act_type = "relu")
net6 = mx.sym.Convolution(data = net5,kernel = (3,3),stride = (2,2),num_filter = 64)
net7 = mx.sym.Activation(data = mx.sym.BatchNorm(net6),act_type = "relu")
net8 = mx.sym.Convolution(data = mx.sym.add_n(net7,net3),kernel = (3,3),stride = (2,2),num_filter = 64)
net9 = mx.sym.Activation(data = mx.sym.BatchNorm(net8),act_type = "relu")
net10 = mx.sym.Convolution(data = net9,kernel = (3,3),stride = (2,2),num_filter = 64)
net11 = mx.sym.Activation(data = mx.sym.BatchNorm(net10),act_type = "relu")
net12 = mx.sym.Convolution(data = mx.sym.add_n(net11,net7),kernel = (3,3),stride = (2,2),num_filter = 64)
net13 = mx.sym.Activation(data = mx.sym.BatchNorm(net12),act_type = "relu")
net14 = mx.sym.Convolution(data = net13,kernel = (3,3),stride = (2,2),num_filter = 64)
net15 = mx.sym.Activation(data = mx.sym.BatchNorm(net14),act_type = "relu")
net16 = mx.sym.Convolution(data = mx.sym.add_n(net15,net11),kernel = (3,3),stride = (2,2),num_filter = 128)
net17 = mx.sym.Activation(data = mx.sym.BatchNorm(net16),act_type = "relu")
net18 = mx.sym.Convolution(data = net17,kernel = (3,3),stride = (2,2),num_filter = 128)
net19 = mx.sym.Activation(data = mx.sym.BatchNorm(net18),act_type = "relu")
net20 = mx.sym.Convolution(data = mx.sym.add_n(
	mx.sym.Convolution(data = net15,num_filter = 128,kernel = (1,1),stride = (1,1))
	net19),kernel = (3,3),stride = (2,2),num_filter = 128)
net21 = mx.sym.Activation(data = mx.sym.BatchNorm(net20),act_type = "relu")
net22 = mx.sym.Convolution(data = net21,kernel = (3,3),stride = (2,2),num_filter = 128)
net23 = mx.sym.Activation(data = mx.sym.BatchNorm(net22),act_type = "relu")
net24 = mx.sym.Convolution(data = mx.sym.add_n(net23,net19),kernel = (3,3),stride = (2,2),num_filter = 128)
net25 = mx.sym.Activation(data = mx.sym.BatchNorm(net24),act_type = "relu")
net26 = mx.sym.Convolution(data = net25,kernel = (3,3),stride = (2,2),num_filter = 128)
net27 = mx.sym.Activation(data = mx.sym.BatchNorm(net26),act_type = "relu")
net28 = mx.sym.Convolution(data = mx.sym.add_n(net27,net23),kernel = (3,3),stride = (2,2),num_filter = 128)
net29 = mx.sym.Activation(data = mx.sym.BatchNorm(net28),act_type = "relu")
net30 = mx.sym.Convolution(data = net29,kernel = (3,3),stride = (2,2),num_filter = 128)
net31 = mx.sym.Activation(data = mx.sym.BatchNorm(net30),act_type = "relu")
net32 = mx.sym.Convolution(data = mx.sym.add_n(net31,net27),kernel = (3,3),stride = (2,2),num_filter = 256)
net33 = mx.sym.Activation(data = mx.sym.BatchNorm(net32),act_type = "relu")
net34 = mx.sym.Convolution(data = net33,kernel = (3,3),stride = (2,2),num_filter = 256)
net35 = mx.sym.Activation(data = mx.sym.BatchNorm(net34),act_type = "relu")
net36 = mx.sym.Convolution(data = mx.sym.add_n(
	net35,
	mx.sym.Convolution(data = net31,kernel = (1,1),num_filter = 256,stride = (1,1))),kernel = (3,3),stride = (2,2),num_filter = 256)
net37 = mx.sym.Activation(data = mx.sym.BatchNorm(net36),act_type = "relu")
net38 = mx.sym.Convolution(data = net37,kernel = (3,3),stride = (2,2),num_filter = 256)
net39 = mx.sym.Activation(data = mx.sym.BatchNorm(net38),act_type = "relu")
net40 = mx.sym.Convolution(data = mx.sym.add_n(net39,net35),kernel = (3,3),stride = (2,2),num_filter = 256)
net41 = mx.sym.Activation(data = mx.sym.BatchNorm(net40),act_type = "relu")
net42 = mx.sym.Convolution(data = net41,kernel = (3,3),stride = (2,2),num_filter = 256)
net43 = mx.sym.Activation(data = mx.sym.BatchNorm(net42),act_type = "relu")
net44 = mx.sym.Convolution(data = mx.sym.add_n(net43,net39),kernel = (3,3),stride = (2,2),num_filter = 256)
net45 = mx.sym.Activation(data = mx.sym.BatchNorm(net44),act_type = "relu")
net46 = mx.sym.Convolution(data = net45,kernel = (3,3),stride = (2,2),num_filter = 256)
net47 = mx.sym.Activation(data = mx.sym.BatchNorm(net46),act_type = "relu")
net48 = mx.sym.Convolution(data = mx.sym.add_n(net47,net43),kernel = (3,3),stride = (2,2),num_filter = 256)
net49 = mx.sym.Activation(data = mx.sym.BatchNorm(net48),act_type = "relu")
net50 = mx.sym.Convolution(data = net49,kernel = (3,3),stride = (2,2),num_filter = 256)
net51 = mx.sym.Activation(data = mx.sym.BatchNorm(net50),act_type = "relu")
net52 = mx.sym.Convolution(data = mx.sym.add_n(net51,net47),kernel = (3,3),stride = (2,2),num_filter = 256)
net53 = mx.sym.Activation(data = mx.sym.BatchNorm(net52),act_type = "relu")
net54 = mx.sym.Convolution(data = net53,kernel = (3,3),stride = (2,2),num_filter = 256)
net55 = mx.sym.Activation(data = mx.sym.BatchNorm(net54),act_type = "relu")
net56 = mx.sym.Convolution(data = mx.sym.add_n(net51,net55),kernel = (3,3),stride = (2,2),num_filter = 512)
net57 = mx.sym.Activation(data = mx.sym.BatchNorm(net56),act_type = "relu")
net58 = mx.sym.Convolution(data = net57,kernel = (3,3),stride = (2,2),num_filter = 512)
net59 = mx.sym.Activation(data = mx.sym.BatchNorm(net58),act_type = "relu")
net60 = mx.sym.Convolution(data = mx.sym.add_n(
	mx.sym.Convolution(data = net55,kernel = (1,1),num_filter = 512,stride = (1,1))
	net59),kernel = (3,3),stride = (2,2),num_filter = 512)
net61 = mx.sym.Activation(data = mx.sym.BatchNorm(net60),act_type = "relu")
net62 = mx.sym.Convolution(data = net61,kernel = (3,3),stride = (2,2),num_filter = 512)
net63 = mx.sym.Activation(data = mx.sym.BatchNorm(net62),act_type = "relu")
net64 = mx.sym.Convolution(data = mx.sym.add_n(net59,net63),kernel = (3,3),stride = (2,2),num_filter = 512)
net65 = mx.sym.Activation(data = mx.sym.BatchNorm(net64),act_type = "relu")
net66 = mx.sym.Convolution(data = net65,kernel = (3,3),stride = (2,2),num_filter = 512)
net67 = mx.sym.Activation(data = mx.sym.BatchNorm(net66),act_type = "relu")
net68 = mx.sym.Pooling(data = net67,pool_type = "avg",global_pool = True)
net69 = mx.sym.FullyConnected(data = net68,num_hidden = 1000)
net70 = mx.sym.Activation(data = net69,act_type = "relu")
net71 = mx.sym.FullyConnected(data = net70,num_hidden = 10)
net72 = mx.sym.SoftmaxOutput(data = net71)

logging.getLogger().setLevel(logging.DEBUG)

model = mx.model.FeedForward(
    symbol = net72,       # network structure
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