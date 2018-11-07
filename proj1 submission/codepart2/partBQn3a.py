#
# Project 1, starter code part b
#
import datetime
from matplotlib.pyplot import cm
import tensorflow as tf
import numpy as np
import pylab as plt
from sklearn.model_selection import KFold

NUM_FEATURES = 8
learning_rate =5e-6
epochs = 300
batch_size = 32

num_neurons = [20,40,60,80,100]
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
# trainX = trainX[:9000]
# trainY = trainY[:9000]

# Create the model
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
y_ = tf.placeholder(tf.float32, [None, 1])
color=iter(cm.rainbow(np.linspace(0,1,len(num_neurons))))
deltas = []
plt.figure(1)
plt.xlabel("Epochs")
plt.ylabel('Average Cross Validation Errors in RMSE in thousands')
finalResult = []
for num_neuron in num_neurons:
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
	reg_loss = reg_loss*beta
	loss = tf.reduce_mean(tf.square(y_ - y))  + reg_loss

	#Create the gradient descent optimizer with the given learning rate.
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	train_op = optimizer.minimize(loss)
	error = tf.reduce_mean(tf.square(y_ - y))


	def run_train(session, train_x, train_y,val_x,val_y):
		session.run(tf.global_variables_initializer())
		results = []
		for epoch in range(epochs):
			idx = np.arange(train_x.shape[0])
			np.random.shuffle(idx)
			train_x, train_y = train_x[idx], train_y[idx]
			if epoch % 10 == 0:
				print("Now at epoch ",epoch)
			results.append(session.run(error, feed_dict={x: val_x, y_: val_y}))
			total_batch = int(train_x.shape[0] / batch_size)

			for i in range(total_batch):
			  batch_x = train_x[i*batch_size:(i+1)*batch_size]
			  batch_y = train_y[i*batch_size:(i+1)*batch_size]
			  train_op.run(feed_dict={x: batch_x, y_: batch_y})


		return results

	def cross_validate(session, split_size=5):
		results = []
		#test_errors = []
		kf = KFold(n_splits=split_size)

		for train_idx, val_idx in kf.split(trainX, trainY):
			train_x = trainX[train_idx]
			train_y = trainY[train_idx]
			val_x = trainX[val_idx]
			val_y = trainY[val_idx]
			results.append(np.sqrt(run_train(session, train_x, train_y,val_x,val_y)))



		return [sum(x)/5 for x in zip(*results)]

	c=next(color)
	a = datetime.datetime.now()


	with tf.Session() as session:

		cross_vals = cross_validate(session)
		print("now at neuron ",num_neuron)
		plt.figure(1)
		plt.plot([i for i in range(len(cross_vals))],cross_vals,c=c, label=str(num_neuron) + " neurons"  )
		finalResult.append(cross_vals[-1])

	b = datetime.datetime.now()
	delta = b - a
	deltas.append(int(delta.total_seconds()))


plt.figure(1)
plt.legend()

plt.figure(2)
plt.plot(num_neurons,finalResult)
plt.xlabel("num of neurons")
plt.ylabel('Cross Validation Errors in RMSE in thousands')

#plt.figure(2)
#plt.legend()
print(deltas)
plt.show()
plt.savefig("3a.png")
