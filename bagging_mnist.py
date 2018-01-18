from __future__ import print_function
from collections import Counter
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import numpy as np
import seaborn as sns
# Import MNIST data
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist.train.images.shape)
index_in_epoch = 0
epoch_completed =0
num_of_weak = 5
num_of_traindata = mnist.train.images.shape[0]
num_of_testdata = mnist.test.images.shape[0]
train_data = np.zeros((num_of_weak,num_of_traindata,784))
train_label = np.zeros((num_of_weak,num_of_traindata,10))

_train_data = np.zeros((num_of_weak,num_of_traindata,784))
_train_label = np.zeros((num_of_weak,num_of_traindata,10))

for i in range(num_of_weak):
    index = np.random.randint(0,num_of_traindata-1,(1,num_of_traindata))
    train_data[i] =mnist.train.images[index]
    train_label[i] = mnist.train.labels[index]
learning_rate = 0.0001
num_steps = 1000
batch_size = 128
display_step = 100

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


def next_batch():
    global index_in_epoch
    global  epoch_completed
    print(index_in_epoch)
    print(num_of_traindata)
    start = index_in_epoch
    if index_in_epoch ==0 and epoch_completed ==0 :
        for data_index in range(num_of_weak):
            perm = np.arange(num_of_traindata)
            np.random.shuffle(perm)
            _train_data[data_index] = train_data[data_index,perm]
            _train_label[data_index] = train_label[data_index, perm]
    if start +batch_size >num_of_traindata:
        epoch_completed += 1
        rest_num_examples = num_of_traindata - start
        image_rest_part = _train_data[:,start:num_of_traindata]
        label_rest_part= _train_label[:, start:num_of_traindata]

        for data_index in range(num_of_weak):
            perm = np.arange(num_of_traindata)
            np.random.shuffle(perm)
            _train_data[data_index] = train_data[data_index,perm]
            _train_label[data_index] = train_label[data_index, perm]
        start = 0
        index_in_epoch = batch_size-rest_num_examples
        end = index_in_epoch
        image_new_part = _train_data[:,start:end]
        label_new_part = _train_label[:,start:end]

        return np.concatenate((image_rest_part,image_new_part),axis =1),np.concatenate((label_rest_part,label_new_part),axis=1)

    else:
        index_in_epoch +=batch_size
        end = index_in_epoch
        return _train_data[:,start:end],_train_label[:,start:end]


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # # Convolution Layer
    # conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # # Max Pooling (down-sampling)
    # conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv1, [-1, weights['out'].get_shape().as_list()[0]])
    # fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    # fc1 = tf.nn.relu(fc1)
    # # Apply Dropout
    # fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

def create_graph():


    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.constant(np.array(np.random.normal(0,0.1,[5,5,1,64]),np.float32)),

        'out': tf.Variable(np.array(np.random.normal(0,0.1,[14*14*64,num_classes]),np.float32))
    }

    biases = {
        'bc1': tf.constant(np.array(np.random.normal(0,0.1,[64]),np.float32)),
        'out': tf.Variable(np.array(np.random.normal(0,0.1,[num_classes]),np.float32))
    }

    # Construct model
    logits = conv_net(X, weights, biases, keep_prob)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)


    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return weights,biases,loss_op,train_op,prediction,accuracy
weights,biases,loss_op,train_op,prediction,accuracy =create_graph()
weights2,biases2,loss_op2,train_op2,prediction2,accuracy2 =create_graph()
weights3,biases3,loss_op3,train_op3,prediction3,accuracy3 =create_graph()
weights4,biases4,loss_op4,train_op4,prediction4,accuracy4 =create_graph()
weights5,biases5,loss_op5,train_op5,prediction5,accuracy5 =create_graph()

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    start = time.clock()
    e_loss=0
    e_acc =0

    for step in range(1, num_steps+1):
        batch_x, batch_y = next_batch()
        # Run optimization op (backprop)
        _,t_loss1,t_acc1 = sess.run([train_op,loss_op,accuracy], feed_dict={X: batch_x[0], Y: batch_y[0], keep_prob: 0.5})
        _, t_loss2, t_acc2 = sess.run([train_op2, loss_op2, accuracy2], feed_dict={X: batch_x[1], Y: batch_y[1], keep_prob: 0.5})
        _, t_loss3, t_acc3 = sess.run([train_op3, loss_op3, accuracy3], feed_dict={X: batch_x[2], Y: batch_y[2], keep_prob: 0.5})
        _, t_loss4, t_acc4 = sess.run([train_op4, loss_op4, accuracy4], feed_dict={X: batch_x[3], Y: batch_y[3], keep_prob: 0.5})
        _, t_loss5, t_acc5 = sess.run([train_op5, loss_op5, accuracy5], feed_dict={X: batch_x[4], Y: batch_y[4], keep_prob: 0.5})

        if step % 1000 == 0 :
            # Calculate batch loss and accuracy
            train_loss1, train_acc1 = sess.run([loss_op, accuracy], feed_dict={X: batch_x[0],
                                                                 Y: batch_y[0],
                                                                 keep_prob: 1.0})
            train_loss2, train_acc2 = sess.run([loss_op2, accuracy2], feed_dict={X: batch_x[1],
                                                                               Y: batch_y[1],
                                                                               keep_prob: 1.0})
            train_loss3, train_acc3 = sess.run([loss_op3, accuracy3], feed_dict={X: batch_x[2],
                                                                               Y: batch_y[2],
                                                                               keep_prob: 1.0})
            train_loss4, train_acc4 = sess.run([loss_op4, accuracy4], feed_dict={X: batch_x[3],
                                                                                 Y: batch_y[3],
                                                                                 keep_prob: 1.0})
            train_loss5, train_acc5 = sess.run([loss_op5, accuracy5], feed_dict={X: batch_x[4],
                                                                                 Y: batch_y[4],
                                                                                 keep_prob: 1.0})

            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(train_loss1) + ", Training Accuracy= " + \
                  "{:.3f}".format(train_acc1))
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(train_loss2) + ", Training Accuracy= " + \
                  "{:.3f}".format(train_acc2))
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(train_loss3) + ", Training Accuracy= " + \
                  "{:.3f}".format(train_acc3))
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(train_loss4) + ", Training Accuracy= " + \
                  "{:.3f}".format(train_acc4))
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(train_loss5) + ", Training Accuracy= " + \
                  "{:.3f}".format(train_acc5))
            acc =0
            loss=0
            test_acc=[0,0,0,0,0,0]
            test_loss=[0,0,0,0,0]
            for i in range(10):
                pred=[]
                start_index = i * 1000
                end_index = (i + 1) * 1000
                pred1,acc,loss= sess.run([prediction,accuracy,loss_op], feed_dict={X: mnist.test.images[start_index:end_index],
                                                     Y: mnist.test.labels[start_index:end_index],
                                                     keep_prob: 1.0})
                test_loss[0]+=loss
                test_acc[0]+=acc
                pred2,acc, loss = sess.run([prediction2,accuracy2, loss_op2], feed_dict={X: mnist.test.images[start_index:end_index],
                                                                     Y: mnist.test.labels[start_index:end_index],
                                                                     keep_prob: 1.0})
                test_loss[1] += loss
                test_acc[1] += acc

                pred3,acc, loss = sess.run([prediction3,accuracy3, loss_op3], feed_dict={X: mnist.test.images[start_index:end_index],
                                                                       Y: mnist.test.labels[start_index:end_index],
                                                                       keep_prob: 1.0})
                test_loss[2] += loss
                test_acc[2] += acc

                pred4, acc, loss = sess.run([prediction4, accuracy4, loss_op4],
                                            feed_dict={X: mnist.test.images[start_index:end_index],
                                                       Y: mnist.test.labels[start_index:end_index],
                                                       keep_prob: 1.0})
                test_loss[3] += loss
                test_acc[3] += acc

                pred5, acc, loss = sess.run([prediction5, accuracy5, loss_op5],
                                            feed_dict={X: mnist.test.images[start_index:end_index],
                                                       Y: mnist.test.labels[start_index:end_index],
                                                       keep_prob: 1.0})
                test_loss[4] += loss
                test_acc[4] += acc

                pred1 = np.argmax(pred1,1)
                pred2 = np.argmax(pred2,1)
                pred3 = np.argmax(pred3, 1)
                pred4 = np.argmax(pred4, 1)
                pred5 = np.argmax(pred5, 1)

                pred.append(pred1)
                pred.append(pred2)
                pred.append(pred3)
                pred.append(pred4)
                pred.append(pred5)

                pred = np.array(pred)
                temp = np.zeros((1000))
                for j in range(1000):
                    word_counts = Counter(pred[:,j])
                    temp[j] = word_counts.most_common(1)[0][0]

                acc = np.mean(np.equal(temp,np.argmax(mnist.test.labels[start_index:end_index],1)))
                # acc ,temppred= sess.run([end_accuracy,end_pred], feed_dict={X: mnist.test.images[start_index:end_index],
                #                                                        Y: mnist.test.labels[start_index:end_index],
                #                                                        keep_prob: 1.0})
                # print(temppred.shape)
                test_acc[5] += acc

            print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.6f}".format(test_loss[0] / 10) + ", test Accuracy= " + \
                      "{:.6f}".format(test_acc[0] / 10) + "  time= " + "{:.6f}".format(time.clock() - start))
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(test_loss[1] / 10) + ", test Accuracy= " + \
                  "{:.6f}".format(test_acc[1] / 10) + "  time= " + "{:.6f}".format(time.clock() - start))
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(test_loss[2] / 10) + ", test Accuracy= " + \
                  "{:.6f}".format(test_acc[2] / 10) + "  time= " + "{:.6f}".format(time.clock() - start))
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(test_loss[3] / 10) + ", test Accuracy= " + \
                  "{:.6f}".format(test_acc[3] / 10) + "  time= " + "{:.6f}".format(time.clock() - start))
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(test_loss[4] / 10) + ", test Accuracy= " + \
                  "{:.6f}".format(test_acc[4] / 10) + "  time= " + "{:.6f}".format(time.clock() - start))
            print("end_acc :%.6f"%(test_acc[5]/10))
    print("cost_time: %.6f"% (time.clock() - start))
    print("Optimization Finished!")







