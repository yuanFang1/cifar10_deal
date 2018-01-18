import cifar10
import cifar10_input
import tensorflow as tf
import numpy as np
import time
from collections import Counter
max_steps =500
batch_size =128
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'

cifar10.maybe_download_and_extract()

images_train,labels_train = cifar10_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)

images_test ,labels_test = cifar10_input.inputs(eval_data=True, data_dir= data_dir,batch_size=batch_size)

image_holder = tf.placeholder(tf.float32,[batch_size,24,24,3])
label_holder = tf.placeholder(tf.int32,[batch_size])


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

def conv2d(x,W,b,strides=1):
    x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    return x
def maxpool2d(x,k=2):
    return tf.nn.max_pool(x,ksize=[1,k+1,k+1,1],strides=[1,k,k,1],padding='SAME')
def conv_net(image_holder,weights,biases):
    conv1 = tf.nn.relu(conv2d(image_holder,weights['wc1'],biases['bc1']))
    pool1 = maxpool2d(conv1)
    #
    # conv2 = tf.nn.relu(conv2d(pool1,weights['wc2'],biases['bc2']))
    # pool2 = maxpool2d(conv2)

    fc1 = tf.reshape(pool1, [-1, weights['out'].get_shape().as_list()[0]])
    # fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    # fc1 = tf.nn.relu(fc1)

    # fc1 = tf.nn.dropout(fc1, dropout)

    # fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    # fc2 = tf.nn.relu(fc2)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

def create_graph():
    weights = {
        'wc1': constant_variable_weight([5,5,3,64],stddev=0.1,is_var=0),
        # 'wc2': constant_variable_weight([5,5,32,64],stddev=5e-2),
        # 'wd1': constant_variable_weight([12*12*32,1024],stddev=0.04),
        # # 'wd2': constant_with_weight_loss([384,192],stddev=0.04),
        'out': constant_variable_weight([12*12*64,10],stddev=1/192.0),
    }

    biases = {
        'bc1': constant_variable_biases(0.0,shape=[64],is_var=0),
        # 'bc2': constant_variable_biases(0.1,shape=[64]),
        # 'bd1': constant_variable_biases(0.1,shape=[1024]),
        # # 'bd2': constant_variable_biases(0.1,shape=[192]),
        'out': constant_variable_biases(0.0,shape=[10])
    }

    logits = conv_net(image_holder,weights,biases)
    pred = tf.argmax(tf.nn.softmax(logits),1)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label_holder,
                                                                   name='cross_entropy_per_example')
    loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
    top_k_op = tf.nn.in_top_k(logits,label_holder,1)
    return pred,weights,biases,loss,train_op,top_k_op
pred,weights,biases,loss,train_op,top_k_op =create_graph()
pred2,weights2,biases2,loss2,train_op2,top_k_op2 =create_graph()
pred3,weights3,biases3,loss3,train_op3,top_k_op3 =create_graph()
pred4,weights4,biases4,loss4,train_op4,top_k_op4 =create_graph()
pred5,weights5,biases5,loss5,train_op5,top_k_op5 =create_graph()

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()


#训练开始
train_time = time.time()
for step in range(max_steps):
    start_time = time.time()
    image_batch,label_batch = sess.run([images_train,labels_train])
    image_batch2,label_batch2 = sess.run([images_train,labels_train])
    image_batch3,label_batch3 = sess.run([images_train,labels_train])
    image_batch4,label_batch4 = sess.run([images_train,labels_train])
    image_batch5,label_batch5 = sess.run([images_train,labels_train])
    _,loss_value =sess.run([train_op,loss],feed_dict={image_holder:image_batch,label_holder:label_batch})
    _,loss_value2 =sess.run([train_op2,loss2],feed_dict={image_holder:image_batch2,label_holder:label_batch2})
    _,loss_value3 =sess.run([train_op3,loss3],feed_dict={image_holder:image_batch3,label_holder:label_batch3})
    _,loss_value4 =sess.run([train_op4,loss4],feed_dict={image_holder:image_batch4,label_holder:label_batch4})
    _,loss_value5 =sess.run([train_op5,loss5],feed_dict={image_holder:image_batch5,label_holder:label_batch5})

    duration = time.time() - start_time
    if step %100 == 0:
        examples_per_sec = batch_size/duration
        sec_per_batch = float(duration)
        format_str = ('step %d,loss=%.2f(%.1f examples/sec; %.3f sec/batch')
        print(format_str % (step,loss_value,examples_per_sec,sec_per_batch))
        print(format_str % (step,loss_value2,examples_per_sec,sec_per_batch))
        print(format_str % (step,loss_value3,examples_per_sec,sec_per_batch))
        print(format_str % (step,loss_value4,examples_per_sec,sec_per_batch))
        print(format_str % (step,loss_value5,examples_per_sec,sec_per_batch))



print('train_time: ',time.time() - train_time)

#测试开始
num_examples =10000
import math
num_iter = int(math.ceil(num_examples/batch_size))
true_count =[0,0,0,0,0,0]
total_sample_count = num_iter *batch_size
step =0
while step < num_iter:

    image_batch,label_batch = sess.run([images_test,labels_test])
    predictions,acc = sess.run([pred,top_k_op],feed_dict={image_holder:image_batch,label_holder:label_batch})
    predictions2,acc2 = sess.run([pred2,top_k_op2],feed_dict={image_holder:image_batch,label_holder:label_batch})
    predictions3,acc3 = sess.run([pred3,top_k_op3],feed_dict={image_holder:image_batch,label_holder:label_batch})
    predictions4,acc4 = sess.run([pred4,top_k_op4],feed_dict={image_holder:image_batch,label_holder:label_batch})
    predictions5,acc5 = sess.run([pred5,top_k_op5],feed_dict={image_holder:image_batch,label_holder:label_batch})
    temppred =[]
    temppred.append(predictions)
    temppred.append(predictions2)
    temppred.append(predictions3)
    temppred.append(predictions4)
    temppred.append(predictions5)
    #获得投票结果
    temppred = np.array(temppred)
    temp =np.zeros(batch_size)
    for j in range(batch_size):
        word_counts = Counter(temppred[:,j])
        temp[j] = word_counts.most_common(1)[0][0]

    acc6 = np.equal(temp,label_batch)
    true_count[0]+=np.sum(acc)
    true_count[1]+=np.sum(acc2)
    true_count[2]+=np.sum(acc3)
    true_count[3]+=np.sum(acc4)
    true_count[4]+=np.sum(acc5)
    true_count[5]+=np.sum(acc6)
    step +=1
precision = true_count[0]/total_sample_count
precision2 = true_count[1]/total_sample_count
precision3 = true_count[2]/total_sample_count
precision4 = true_count[3]/total_sample_count
precision5 = true_count[4]/total_sample_count
end_pred = true_count[5]/total_sample_count
print('precision @ 1 =%.5f' % precision)
print('precision @ 1 =%.5f' % precision2)
print('precision @ 1 =%.5f' % precision3)
print('precision @ 1 =%.5f' % precision4)
print('precision @ 1 =%.5f' % precision5)
print('precision @ 1 =%.5f' % end_pred)
