#coding:utf-8
import tensorflow as tf
import numpy as np
import tensorflow.contrib as tf_contrib
import random
import os
from math import ceil
import scipy.misc as misc
from skimage import color
from tensorflow.python.training import moving_averages
def onehot(label,class_num):
    labels = np.zeros([len(label), class_num])
    labels[np.arange(len(label)), label] = 1
    labels = np.reshape(labels, [-1, class_num])
    return labels
def shuffle(train_samples,train_x,train_y):
    # shuffle the dataset
    #print(train_y)
    #print(len(train_y))
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    perm = np.arange(train_samples)
    random.shuffle(perm)
    #print(perm)
    x_train= train_x[np.array(perm)]
    y_train = train_y[np.array(perm)]
    return x_train,y_train
def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):  ##循环每次随机为True or False
            batch[i] = np.fliplr(batch[i])
    return batch
def read_images(batch_image):
    batch_x=[]
    for img in batch_image:
        img_arr=misc.imread(img)
        batch_x.append(img_arr/255.0)
    return np.array(batch_x)
def load_task(data_path,label_path):
    task_num=os.listdir(label_path)
    train_task,test_task=[],[]
    for i in task_num:
        train_task_imgs, train_task_labels, test_task_imgs, test_task_labels = [], [], [], []
        task_path=os.path.join(label_path,i)
        if i[:5]=='train':
            f_train = open(task_path)
            for train_img in f_train.readlines():
                path_1=os.path.join(data_path,train_img.strip())
                train_task_imgs.append(path_1.split('\t')[0])
                train_task_labels.append(int(path_1.split('\t')[1]))
            train_task.append((train_task_imgs,train_task_labels))

        else:
            f_test = open(task_path)
            for test_img in f_test.readlines():
                path_2=os.path.join(data_path,test_img.strip())
                test_task_imgs.append(path_2.split('\t')[0])
                test_task_labels.append(int(path_2.split('\t')[1]))
            test_task.append((test_task_imgs,test_task_labels))
    return train_task,test_task

def residual_block_first(x,is_training, w1, w2,w3, b1, b2, strides1,strides2,name1,name2,name3,pad="SAME"):
    in_channel = x.get_shape().as_list()[-1]
    # Shortcut connection

    if in_channel == w2.get_shape().as_list()[-1]:
        if strides1[1] == 1:
            shortcut = tf.identity(x)
        else:
            shortcut = tf.nn.max_pool(x, [1,2,2,1], [1,2,2,1], 'VALID')
    else:
        shortcut = tf.nn.conv2d(x, w3,strides=[1,2,2,1],padding="SAME",name='shortcut')
        shortcut=batch_normlize(shortcut,is_training,scope=name1)

    conv_1 = tf.nn.conv2d(x, w1, strides=strides1, padding=pad) + b1
    bn_1 = tf.nn.relu(batch_normlize(conv_1,is_training,scope=name2))

    conv_2 = tf.nn.conv2d(bn_1, w2, strides=strides2, padding=pad) + b2
    bn_2 = batch_normlize(conv_2,is_training,scope=name3)

    out = tf.nn.relu(shortcut + bn_2)
    return out
def residual_block(x,is_training, w1, w2, b1, b2, strides1,strides2,name1,name2,pad="SAME"):
    shorcut = x
    conv_1 = tf.nn.conv2d(x, w1, strides=strides1, padding=pad) + b1
    bn_1 = tf.nn.relu(batch_normlize(conv_1,is_training,scope=name1))
    conv_2 = tf.nn.conv2d(bn_1, w2, strides=strides2, padding=pad) + b2
    bn_2 = batch_normlize(conv_2,is_training,scope=name2)
    out = tf.nn.relu(shorcut + bn_2)
    return out

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
    return gap
def weight_init(shape,name):
    with tf.name_scope('weights'):
        return tf.get_variable(name=name,shape=shape,initializer=tf.contrib.layers.xavier_initializer())
def bias_variable(shape):
    with tf.name_scope('bias'):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
def batch_normlize(x,is_training,scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=tf.GraphKeys.UPDATE_OPS,
                                        is_training=is_training, scope=scope)


class Model():
    def __init__(self,x,is_training,dropout):
        self.x = x  # input placeholder
        self.is_training=is_training
        self.dropout=dropout
        ##conv1-bn-pool
        self.w1 = weight_init(shape=[7, 7, 3, 64],name='w1')
        self.b1 = bias_variable(shape=[64])
        conv1 = tf.nn.conv2d(x, self.w1, strides=[1, 2, 2, 1], padding="SAME") + self.b1
        conv1_bn = tf.nn.relu(batch_normlize(conv1,self.is_training,scope='conv1'))
        pool1 = tf.nn.max_pool(conv1_bn, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        # conv2_x
        # conv2_1
        self.w2 = weight_init(shape=[3, 3, 64, 64],name='w2')
        self.b2 = bias_variable(shape=[64])
        self.w3 = weight_init(shape=[3, 3, 64, 64],name='w3')
        self.b3 = bias_variable(shape=[64])
        conv2_1 = residual_block(pool1,self.is_training, self.w2, self.w3, self.b2,self.b3, strides1=[1, 1, 1, 1], strides2=[1, 1, 1, 1],name1='conv2_1_1',name2='conv2_1_2')

        ##conv2_2
        self.w4 = weight_init(shape=[3, 3, 64, 64],name='w4')
        self.b4 = bias_variable(shape=[64])
        self.w5 = weight_init(shape=[3, 3, 64, 64],name='w5')
        self.b5 = bias_variable(shape=[64])
        conv2_2 = residual_block(conv2_1,self.is_training,self.w4, self.w5, self.b4, self.b5, strides1=[1, 1, 1, 1], strides2=[1, 1, 1, 1],name1='conv2_2_1',name2='conv2_2_2')

        ##conv3_1
        self.w6 = weight_init(shape=[1, 1, 64, 128],name='w6')
        self.w7 = weight_init(shape=[3, 3, 64, 128],name='w7')
        self.b7 = bias_variable(shape=[128])
        self.w8 = weight_init(shape=[3, 3, 128, 128],name='w8')
        self.b8 = bias_variable(shape=[128])
        conv3_1 = residual_block_first(conv2_2,self.is_training,self.w7, self.w8, self.w6, self.b7, self.b8, strides1=[1, 2, 2, 1], strides2=[1, 1, 1, 1],name1='conv3_1_1',name2='conv3_1_2',name3='conv3_1_3')

        ##conv3_2
        self.w9 = weight_init(shape=[3, 3, 128, 128],name='w9')
        self.b9 = bias_variable(shape=[128])
        self.w10 = weight_init(shape=[3, 3, 128, 128],name='w10')
        self.b10 = bias_variable(shape=[128])
        conv3_2 = residual_block(conv3_1,self.is_training,self.w9, self.w10, self.b9,self.b10, strides1=[1, 1, 1, 1], strides2=[1, 1, 1, 1],name1='conv3_2_1',name2='conv3_2_2')

        # conv4_1
        self.w11 = weight_init(shape=[1, 1, 128, 256],name='w11')
        self.w12 = weight_init(shape=[3, 3, 128, 256],name='w12')
        self.b12 = bias_variable(shape=[256])
        self.w13 = weight_init(shape=[3, 3, 256, 256],name='w13')
        self.b13 = bias_variable(shape=[256])
        conv4_1 = residual_block_first(conv3_2,self.is_training,self.w12, self.w13, self.w11,self.b12, self.b13, strides1=[1, 2, 2, 1], strides2=[1, 1, 1, 1],name1='conv4_1_1',name2='conv4_1_2',name3='conv4_1_3')

        ##conv4_2
        self.w14 = weight_init(shape=[3, 3, 256, 256],name='w14')
        self.b14 = bias_variable(shape=[256])
        self.w15 = weight_init(shape=[3, 3, 256, 256],name='w15')
        self.b15 = bias_variable(shape=[256])
        conv4_2 = residual_block(conv4_1,self.is_training,self.w14, self.w15, self.b14,self.b15, strides1=[1, 1, 1, 1], strides2=[1, 1, 1, 1],name1='conv4_2_1',name2='conv4_2_2')

        ##conv5_1
        self.w16 = weight_init(shape=[1, 1, 256, 512],name='w16')
        self.w17 = weight_init(shape=[3, 3, 256, 512],name='w17')
        self.b17 = bias_variable(shape=[512])
        self.w18 = weight_init(shape=[3, 3, 512, 512],name='w18')
        self.b18 = bias_variable(shape=[512])
        conv5_1 = residual_block_first(conv4_2,self.is_training,self.w17, self.w18, self.w16, self.b17, self.b18, strides1=[1, 2, 2, 1], strides2=[1, 1, 1, 1],name1='conv5_1_1',name2='conv5_1_2',name3='conv5_1_3')

        ##conv5_2
        self.w19 = weight_init(shape=[3, 3, 512, 512],name='w19')
        self.b19 = bias_variable(shape=[512])
        self.w20 = weight_init(shape=[3, 3, 512, 512],name='w20')
        self.b20 = bias_variable(shape=[512])
        conv5_2 = residual_block(conv5_1,self.is_training,self.w19, self.w20, self.b19, self.b20, strides1=[1, 1, 1, 1], strides2=[1, 1, 1, 1],name1='conv5_2_1',name2='conv5_2_2')

        ##avg_pool-flatten-fc
        # pool_6 = tf.nn.avg_pool(conv5_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        pool_6 = global_avg_pooling(conv5_2)
        dim = pool_6.get_shape().as_list()
        flat_dim = dim[1] * dim[2] * dim[3]
        flat_6 = tf.reshape(pool_6, [-1, flat_dim])

        self.w21 = weight_init(shape=[flat_dim,1024],name='w21')
        self.b21 = bias_variable(shape=[1024])
        fc1 = tf.nn.relu(tf.matmul(flat_6, self.w21) + self.b21)
        self.drop1=tf.nn.dropout(fc1,keep_prob=self.dropout)
        self.params=tf.trainable_variables()
        return

def optimizer(lr_start,cost,varlist,steps,epoches):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        learning_rate = tf.train.exponential_decay(lr_start, global_step=steps, decay_steps=epoches, decay_rate=0.9,staircase=True) # lr decay=1
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step=steps, var_list=varlist)
        return train_step,learning_rate

x = tf.placeholder(tf.float32, shape=[None,256,256,3])
is_training=tf.placeholder(tf.bool)
dropout=tf.placeholder(tf.float32)
model = Model(x,is_training,dropout)
# x_clone = tf.placeholder(tf.float32,shape=[None,784])
#model_old = Model(x)
# train and test
#NI:3
num_tasks_to_run = 3 #任务数
#NC:5
#num_tasks_to_run = 5 #任务数
#NIC:15
#num_tasks_to_run = 15 #任务数
num_epochs_per_task = [300,400,500]
minibatch_size = 64
epoches = 20000
learning_lr = 0.001

data_path=r'./CLRS dataset/CLRS'
label_path=r'./CLRS dataset/label/NI/run1'  #NI/NC/NIC
train_task,test_task=load_task(data_path,label_path)
def extend_task_data(task_num):
    label_list_train = []
    img_list_train = []
    for m in range(task_num):
        img_list_train.extend(train_task[m][0])
        label_list_train.extend(train_task[m][1])
    return img_list_train,label_list_train
#print(train_task)
y_= tf.placeholder(tf.float32, shape=[None,25])
out_dim = int(y_.get_shape()[1])
lr_steps = tf.Variable(0)

# expand output layer
W_output = weight_init([1024, out_dim],name='w_out')
b_output = bias_variable([out_dim])
task_output = tf.matmul(model.drop1, W_output) + b_output
task_acc = []
for task in range(num_tasks_to_run):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    sess.run(tf.global_variables_initializer())
        # print "Training task: ",task+1,"/",num_tasks_to_run
    print("\t task ", task + 1)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=task_output))
    train_op,learning_rate = optimizer(lr_start=learning_lr, cost=cost, varlist=[model.params, W_output, b_output],steps=lr_steps,epoches=epoches)
    correct_t = tf.equal(tf.argmax(task_output, 1), tf.argmax(y_,1))
    acc_t = tf.reduce_mean(tf.cast(correct_t, tf.float32))
    sess.run(tf.variables_initializer([lr_steps]))
    stop_loss = 1000000
    total_count = 0
    stop_acc = 0
    acc_count = 0
    for epoch in range(1,num_epochs_per_task[task]+1):
        print("\t Epoch ", epoch)
        loss_all, acc_all, lr_all = [], [], []
        train_data,label_data=extend_task_data(task + 1)
        train_img,train_lab=shuffle(len(train_data),train_data,label_data)
        for i in range(int(len(train_task[task][0]) / minibatch_size) + 1):
            batch_x =train_img[i * minibatch_size:(i + 1) * minibatch_size]
            batch_y =train_lab[i * minibatch_size:(i + 1) * minibatch_size]
            train_batch_x=read_images(batch_x)
            train_batch_img=_random_flip_leftright(train_batch_x)
            train_batch_lab=onehot(batch_y,out_dim)
            op, loss_1,  lr_1 = sess.run([train_op, cost,learning_rate],
                                               feed_dict={x: train_batch_img, y_: train_batch_lab, is_training: True,
                                                          dropout: 0.5})
            acc_1=sess.run(acc_t,feed_dict={x: train_batch_img, y_: train_batch_lab, is_training:False,dropout: 1.0})

            loss_all.append(loss_1)
            acc_all.append(acc_1)
            lr_all.append(lr_1)
        mean_loss = np.mean(loss_all)
        mean_acc = np.mean(acc_all)
        mean_lr = np.mean(lr_all)
        if epoch % 1 == 0:
            print("Epoch " + str(epoch) + ", Minibatch Loss=" + \
                    str(mean_loss) + ", Training Accuracy= " + \
                    str(mean_acc) + ", learning_rate:", str(mean_lr))

    correct_prediction = tf.equal(tf.argmax(task_output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    test_x, test_y = test_task[0][0], test_task[0][1]
    iters = len(test_x) // minibatch_size
    te_acc = []
    for k in range(0, iters):
        test_batch_x = test_x[k * minibatch_size:(k + 1) * minibatch_size]
        test_batch_y = test_y[k * minibatch_size:(k + 1) * minibatch_size]
        test_batch_images = read_images(test_batch_x)
        test_batch_labels = onehot(test_batch_y, out_dim)
        acc_ = sess.run(accuracy, feed_dict={x: test_batch_images, y_: test_batch_labels, is_training: False,dropout:1.0})
        te_acc.append(acc_)
    test_avg = np.mean(te_acc)
    print("Task: ", (task + 1), " \tAccuracy: ", test_avg)
    task_acc.append(test_avg)
    sess.close()
file = open('CLRS_Cumulative_reslution_NI.txt', 'w')
for fp in task_acc:
    file.write(str(fp)+',')
file.close()
