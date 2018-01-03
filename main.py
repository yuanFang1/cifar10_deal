import cifar10
import cifar10_input
import tensorflow as tf
import numpy as np
import time

max_steps =3000
batch_size =128
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'

def constant_variable_biases(value,shape,is_var=1):
    if is_var == 1:
        var = tf.Variable(tf.constant(value,shape=shape))
    else:
        var = tf.constant(value=value,shape=shape)
    return var

def constant_variable_weight(shape,stddev,is_var=1):
    if is_var == 1:#等于1取变量，等于0取常量固定
        var = tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    else:
        var = tf.constant(np.array(np.random.normal(0,stddev,shape),np.float32))
    return var


cifar10.maybe_download_and_extract()

images_train,labels_train = cifar10_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)

images_test ,labels_test = cifar10_input.inputs(eval_data=True, data_dir= data_dir,batch_size=batch_size)

image_holder = tf.placeholder(tf.float32,[batch_size,24,24,3])
label_holder = tf.placeholder(tf.int32,[batch_size])


def conv2d(x,W,b,strides=1):
    x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    return x
def maxpool2d(x,k=2):
    return tf.nn.max_pool(x,ksize=[1,k+1,k+1,1],strides=[1,k,k,1],padding='SAME')
def conv_net(image_holder,weights,biases):
    conv1 = tf.nn.relu(conv2d(image_holder,weights['wc1'],biases['bc1']))
    pool1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(pool1,weights['wc2'],biases['bc2']))
    pool2 = maxpool2d(conv2)

    fc1 = tf.reshape(pool2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # fc1 = tf.nn.dropout(fc1, dropout)

    # fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    # fc2 = tf.nn.relu(fc2)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out
def loss(logits,labels):
    labels = tf.cast(labels,tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
    tf.add_to_collection('losses',cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'),name='total_loss')

weights = {
    'wc1': constant_variable_weight([5,5,3,32],stddev=5e-2),
    'wc2': constant_variable_weight([5,5,32,64],stddev=5e-2),
    'wd1': constant_variable_weight([6*6*64,1024],stddev=0.04),
    # 'wd2': constant_with_weight_loss([384,192],stddev=0.04),
    'out': constant_variable_weight([1024,10],stddev=1/192.0),
}

biases = {
    'bc1': constant_variable_biases(0.0,shape=[32]),
    'bc2': constant_variable_biases(0.1,shape=[64]),
    'bd1': constant_variable_biases(0.1,shape=[1024]),
    # 'bd2': constant_variable_biases(0.1,shape=[192]),
    'out': constant_variable_biases(0.0,shape=[10])
}

logits = conv_net(image_holder,weights,biases)
loss = loss(logits,label_holder)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
top_k_op = tf.nn.in_top_k(logits,label_holder,1)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()


#训练开始
train_time = time.time()
for step in range(max_steps):
    start_time = time.time()
    image_batch,label_batch = sess.run([images_train,labels_train])
    _,loss_value =sess.run([train_op,loss],feed_dict={image_holder:image_batch,label_holder:label_batch})
    duration = time.time() - start_time
    if step %10 == 0:
        examples_per_sec = batch_size/duration
        sec_per_batch = float(duration)
        format_str = ('step %d,loss=%.2f(%.1f examples/sec; %.3f sec/batch')
        print(format_str % (step,loss_value,examples_per_sec,sec_per_batch))

print('train_time: ',time.time() - train_time)

#测试开始
num_examples =10000
import math
num_iter = int(math.ceil(num_examples/batch_size))
true_count =0
total_sample_count = num_iter *batch_size
step =0
while step < num_iter:
    image_batch,label_batch = sess.run([images_test,labels_test])
    predictions = sess.run([top_k_op],feed_dict={image_holder:image_batch,label_holder:label_batch})
    true_count+=np.sum(predictions)
    step +=1
precision = true_count/total_sample_count

print('precision @ 1 =%.5f' % precision)