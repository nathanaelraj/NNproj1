#
# Project 1, starter code part b
#

import tensorflow as tf
import numpy as np
import pylab as plt
from matplotlib.pyplot import cm

NUM_FEATURES = 8

learning_rate = 5e-6
epochs = 400
batch_size = 32
num_neuron = 100
seed = 10
beta = 10e-3
np.random.seed(seed)

#read and divide data into test and train sets
#total 20640 data points
cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
X_data, Y_data = cal_housing[:,:8], cal_housing[:,-1]
Y_data = np.array([x/1000 for x in Y_data])
Y_data = (np.asmatrix(Y_data)).transpose()

idx = np.arange(X_data.shape[0])
np.random.shuffle(idx)
X_data, Y_data = X_data[idx], Y_data[idx]

m = 3* X_data.shape[0] // 10

testX, testY =  X_data[:m], Y_data[:m]
trainX, trainY = X_data[m:], Y_data[m:]

testX = (testX- np.mean(trainX, axis=0))/ np.std(trainX, axis=0)
trainX = (trainX- np.mean(trainX, axis=0))/ np.std(trainX, axis=0)
# # experiment with small datasets
# trainX = trainX[:8000]
# trainY = trainY[:8000]



plt.figure(1)
plt.xlabel('epoch')
plt.ylabel('test Error')
color=iter(cm.rainbow(np.linspace(0,1,3)))
finalResult = []
for layers in range(1,4):
	c=next(color)
	tf.reset_default_graph()

	# Create the model
	x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
	y_ = tf.placeholder(tf.float32, [None, 1])

	if(layers == 1):
		#build hidden layer of 30 neurons w Relu
		weights_1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, num_neuron], stddev=1.0 / np.sqrt(NUM_FEATURES), dtype=tf.float32), name='weights_1')
		biases_1 = tf.Variable(tf.zeros([num_neuron]), dtype=tf.float32, name='biases_1')
		u_1 = tf.add(tf.matmul(x, weights_1), biases_1)
		output_1 = tf.nn.relu(u_1)

		#final output layer
		weights_2 = tf.Variable(tf.truncated_normal([num_neuron, 1], stddev=1.0 / np.sqrt(num_neuron), dtype=tf.float32), name='weights_2')
		biases_2 = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='biases_2')
		y = tf.matmul(output_1, weights_2) + biases_2
		reg_loss = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2)
	if(layers ==2):

		#build hidden layer of 30 neurons w Relu
		weights_1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, num_neuron], stddev=1.0 / np.sqrt(NUM_FEATURES), dtype=tf.float32), name='weights_1')
		biases_1 = tf.Variable(tf.zeros([num_neuron]), dtype=tf.float32, name='biases_1')
		u_1 = tf.add(tf.matmul(x, weights_1), biases_1)
		output_1 = tf.nn.relu(u_1)

		weights_2 = tf.Variable(tf.truncated_normal([num_neuron, num_neuron], stddev=1.0 / np.sqrt(NUM_FEATURES), dtype=tf.float32), name='weights_1')
		biases_2 = tf.Variable(tf.zeros([num_neuron]), dtype=tf.float32, name='biases_1')
		u_2 = tf.add(tf.matmul(output_1, weights_2), biases_2)
		output_2 = tf.nn.relu(u_2)

		#final output layer
		weights_3 = tf.Variable(tf.truncated_normal([num_neuron, 1], stddev=1.0 / np.sqrt(num_neuron), dtype=tf.float32), name='weights_2')
		biases_3 = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='biases_2')
		y = tf.matmul(output_2, weights_3) + biases_3
		reg_loss = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2)+ tf.nn.l2_loss(weights_3)
	if(layers ==3):
		#build hidden layer of 30 neurons w Relu
		weights_1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, num_neuron], stddev=1.0 / np.sqrt(NUM_FEATURES), dtype=tf.float32), name='weights_1')
		biases_1 = tf.Variable(tf.zeros([num_neuron]), dtype=tf.float32, name='biases_1')
		u_1 = tf.add(tf.matmul(x, weights_1), biases_1)
		output_1 = tf.nn.relu(u_1)

		weights_2 = tf.Variable(tf.truncated_normal([num_neuron, num_neuron], stddev=1.0 / np.sqrt(NUM_FEATURES), dtype=tf.float32), name='weights_1')
		biases_2 = tf.Variable(tf.zeros([num_neuron]), dtype=tf.float32, name='biases_1')
		u_2 = tf.add(tf.matmul(output_1, weights_2), biases_2)
		output_2 = tf.nn.relu(u_2)

		weights_3 = tf.Variable(tf.truncated_normal([num_neuron, num_neuron], stddev=1.0 / np.sqrt(NUM_FEATURES), dtype=tf.float32), name='weights_1')
		biases_3 = tf.Variable(tf.zeros([num_neuron]), dtype=tf.float32, name='biases_1')
		u_3 = tf.add(tf.matmul(output_2, weights_3), biases_3)
		output_3 = tf.nn.relu(u_2)

		#final output layer
		weights_4 = tf.Variable(tf.truncated_normal([num_neuron, 1], stddev=1.0 / np.sqrt(num_neuron), dtype=tf.float32), name='weights_2')
		biases_4 = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='biases_2')
		y = tf.matmul(output_3, weights_4) + biases_4
		reg_loss = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2) + tf.nn.l2_loss(weights_3) + tf.nn.l2_loss(weights_4)



	reg_loss = reg_loss*beta
	loss = tf.reduce_mean(tf.square(y_ - y))  + reg_loss

	#Create the gradient descent optimizer with the given learning rate.
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	train_op = optimizer.minimize(loss)
	error = tf.reduce_mean(tf.square(y_ - y))

	test_errors = []
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(epochs):
			idx = np.arange(trainX.shape[0])
			np.random.shuffle(idx)
			trainX, trainY = trainX[idx], trainY[idx]
			#batch gradient descent
			for start, end in zip(range(0, trainX.shape[0], batch_size), range(batch_size, trainX.shape[0], batch_size)):
				train_op.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})



			err = np.sqrt(error.eval(feed_dict={x: testX, y_: testY}))
			test_errors.append(err)
		finalResult.append(test_errors[-1])


	# plot learning curves

	plt.plot(range(len(test_errors)), test_errors,c=c,label=  str(layers)+ ' hidden layer')
plt.legend()
for v in tf.trainable_variables():
	print(v)

plt.figure(2)
print("finalResult",finalResult)
plt.plot([3,4,5],finalResult)
plt.xlabel("num of hidden layers")
plt.ylabel('final test error in RMSE in thousands')

plt.show()
