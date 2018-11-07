import math
import tensorflow as tf
import numpy as np
import pylab as plt
import time


# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

NUM_FEATURES = 36
NUM_CLASSES = 6
NUM_HIDDEN = 10
EPOCHS = 1000
learning_rate = 0.01
NUM_INPUT = 4435
num_neurons = 10
seed = 10
beta = 10 **-6
np.random.seed(seed)

train_input = np.loadtxt('sat_train.txt',delimiter=' ')
trainX, train_Y = train_input[:,:36], train_input[:,-1].astype(int)
trainXmin = np.min(trainX, axis=0)
trainXmax = np.max(trainX, axis=0)
trainX = scale(trainX, trainXmin, trainXmax)
train_Y[train_Y == 7] = 6

trainY = np.zeros((train_Y.shape[0], NUM_CLASSES))
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 #one hot matrix

#read test data
test_input = np.loadtxt('sat_test.txt',delimiter=' ')
testX, test_Y = test_input[:,:36], test_input[:,-1].astype(int)
testX = scale(testX, trainXmin, trainXmax)
test_Y[test_Y == 7] = 6

testY = np.zeros((test_Y.shape[0], NUM_CLASSES))
testY[np.arange(test_Y.shape[0]), test_Y-1] = 1 #one hot matrix


n = trainX.shape[0]


# Create the model


x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
#x is ? by 32
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

# Build the graph for the deep net
#w1 and b1 is the weight/bias to the hidden layer
w1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, NUM_HIDDEN], stddev=1.0/math.sqrt(float(NUM_FEATURES))), name='weightToHidden')
b1  = tf.Variable(tf.zeros([NUM_HIDDEN]), name='biasestohidden')

#w2 and b2 are the weight/bias from hidden to output layer
w2 = tf.Variable(tf.truncated_normal([NUM_HIDDEN, NUM_CLASSES], stddev=1.0/math.sqrt(float(NUM_FEATURES))), name='weightToOutput')
b2  = tf.Variable(tf.zeros([NUM_CLASSES]), name='biasestooutput')

hidden_logits = tf.matmul(x, w1) + b1
hidden_activated = tf.sigmoid(hidden_logits)

output_logits = tf.matmul(hidden_activated, w2) + b2

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=output_logits)

loss = tf.reduce_mean(cross_entropy) + tf.square(tf.norm(w1, ord = 'fro', axis = (-2,-1)))*beta + tf.square(tf.norm(w2, ord = 'fro', axis =(-2,-1)))*beta

# Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

correct_prediction = tf.cast(tf.equal(tf.argmax(output_logits, 1), tf.argmax(y_, 1)), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

durations = []
batch_sizes = [4, 8, 16, 32, 64]
for batch_size in batch_sizes:
    train_errors = []
    test_accs =[]
    repetition_in_one_epoch = int(NUM_INPUT / batch_size)
    idx = np.arange(trainX.shape[0])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        start_time = time.time()
        for i in range(EPOCHS):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]
            start = -1 * batch_size
            end = 0
            for j in range(repetition_in_one_epoch):
                start += batch_size
                end += batch_size
                if end > NUM_INPUT:
                    end = NUM_INPUT
                train_op.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})
                # Calculate test and train accuracies at end of epoch
            train_errors.append(loss.eval(feed_dict={x:trainX, y_:trainY}))
            test_accs.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
            if i % (EPOCHS/4) == 0:
                print('Epoch %d: train error %g'%(i, train_errors[i]))
                print('Epoch %d: test accuracy %g'%(i, test_accs[i]))
        duration = time.time() - start_time
        durations.append(duration/EPOCHS)
        print("Final test accuracy for batch size: ", batch_size, ' : ', test_accs[-1])

        # plot learning curves
    plt.figure('Train Error vs Epochs')
    plt.plot(range(EPOCHS), train_errors, label="Batch size of " + str(batch_size))
    plt.xlabel(str(EPOCHS) + ' epochs')
    plt.ylabel('Train error (Cross Entropy With L2 Regularisation)')
    plt.legend()
    plt.figure('Test Accuracy vs Epochs')
    plt.plot(range(EPOCHS), test_accs, label="Batch size of " + str(batch_size))
    plt.xlabel(str(EPOCHS) + ' epochs')
    plt.ylabel('Test accuracy')
    plt.legend()

plt.figure('Average Time Taken for one epoch vs Batch Size')
plt.plot(batch_sizes, durations)
plt.xlabel('Batch sizes')
plt.ylabel('Average Time')

plt.show()
