import numpy as np
from skimage import io
import os as os
import tensorflow as tf

total_epochs = 10
batch_size = 320


def read_index(path):
    '''
    to read the index
    data have lable
    '''
    path = path + '.txt'
    file = open(path)  #
    try:
        file_context = file.read()
        #  file_context = open(file).read().splitlines()

    finally:
        file.close()
    file_context = file_context.replace('\t', '\n').split('\n')
    if len(file_context) % 2 == 1:
        len_of_context = len(file_context) - 1
    else:
        len_of_context = len(file_context)

    x_index = list([])
    y = list([])
    for i in range(len_of_context):
        if i % 2 == 0:
            x_index.append(file_context[i])
        else:
            y.append(file_context[i])
    return x_index, y


def true_name(all_name, local_name):
    a = False
    for name in all_name:
        if name[0:8] == local_name:
            a = True
            return name, a
    return local_name, a


def isblack(data):
    if sum(sum(data)) < 20:
        return True
    else:
        return False


def get_picth(all_name, x_train_index, patch_size_1, patch_size_2, y):
    x_patch = []
    x_patch_index = []
    x_train_use_or_not = []
    x_patch_to_image = []
    y_patch = []
    image_size = []
    for i in range(len(x_train_index)):
        x_index_local, have_index = true_name(all_name, x_train_index[i])
        if have_index:
            x_index_local = './Dataset_A/data/' + x_index_local
            pic_raw = io.imread(x_index_local)

            d1 = int(pic_raw.shape[0] / patch_size_1)
            d2 = int(pic_raw.shape[1] / patch_size_2)

            for patch_i in range(d1):
                for patch_j in range(d2):
                    picture_patch = pic_raw[patch_i * 100:patch_i * 100 + 100, patch_j * 100:patch_j * 100 + 100]
                    x_patch.append(picture_patch)
                    picure_index = [patch_i, patch_j]
                    x_patch_index.append(picure_index)
                    x_patch_to_image.append(i)
                    y_patch.append(y[i])
                    image_size.append(pic_raw.shape)
                    if (isblack(picture_patch)):
                        x_train_use_or_not.append(False)
                    else:
                        x_train_use_or_not.append(True)

    return x_patch, x_patch_index, x_train_use_or_not, x_patch_to_image, y_patch, image_size


def get_data_in_patch(all_name, x_train_index, patch_size_1, patch_size_2, y):
    # delete black one
    x_patch, x_patch_index, x_train_use, image_index, y2, image_size = get_picth(all_name, x_train_index, patch_size_1,
                                                                                 patch_size_2, y)
    final_x_patch = []
    final_x_patch_index = []
    final_image_index = []
    final_y = []
    final_image_size = []
    for i in range(len(x_train_use)):
        if (x_train_use[i]):
            final_x_patch.append(x_patch[i])
            final_x_patch_index.append(x_patch_index[i])
            final_image_index.append(image_index[i])
            final_y.append(y2[i])
            final_image_size.append(image_size[i])
    return final_x_patch, final_x_patch_index, final_image_index, final_y, final_image_size

with tf.device('/gpu:0'):

  x = tf.placeholder(tf.float32, shape=[None, 100, 100, 1], name='x')
  y_true = tf.placeholder(tf.float32, shape=[None, 2], name='y_true')
  y_true_cls = tf.argmax(y_true, 1)

  filter_size = tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.05))
  layer = tf.nn.conv2d(input=x, filter=filter_size, strides=[1, 1, 1, 1], padding='SAME')
  layer = tf.nn.bias_add(layer, tf.Variable(tf.constant(0.05, shape=[32])))
  layer = tf.nn.relu(layer)
  layer = tf.nn.local_response_normalization(layer)
  layer = tf.nn.max_pool(value=layer, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

  filter_size = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.05))
  layer = tf.nn.conv2d(input=layer, filter=filter_size, strides=[1, 1, 1, 1], padding='SAME')
  layer = tf.nn.bias_add(layer, tf.Variable(tf.constant(0.05, shape=[64])))
  layer = tf.nn.relu(layer)
  layer = tf.nn.local_response_normalization(layer)
  layer = tf.nn.max_pool(value=layer, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

  filter_size = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.05))
  layer = tf.nn.conv2d(input=layer, filter=filter_size, strides=[1, 1, 1, 1], padding='SAME')
  layer = tf.nn.bias_add(layer, tf.Variable(tf.constant(0.05, shape=[128])))
  layer = tf.nn.relu(layer)

  filter_size = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.05))
  layer = tf.nn.conv2d(input=layer, filter=filter_size, strides=[1, 1, 1, 1], padding='SAME')
  layer = tf.nn.bias_add(layer, tf.Variable(tf.constant(0.05, shape=[128])))
  layer = tf.nn.relu(layer)
  layer = tf.nn.max_pool(value=layer, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

  weight = tf.Variable(tf.random_normal([13 * 13 * 128, 2]))
  bias = tf.Variable(tf.random_normal([2]))
  layer = tf.reshape(layer, [-1, weight.get_shape().as_list()[0]])
  layer = tf.nn.relu(tf.add(tf.matmul(layer, weight), bias))
  layer = tf.nn.dropout(layer, keep_prob=0.50)

  weight = tf.Variable(tf.random_normal([2, 2]))
  bias = tf.Variable(tf.random_normal([2]))
  layer = tf.reshape(layer, [-1, weight.get_shape().as_list()[0]])
  layer = tf.nn.relu(tf.add(tf.matmul(layer, weight), bias))
  layer = tf.nn.dropout(layer, keep_prob=0.50)

  y_pred = tf.nn.softmax(layer, name='y_pred')
  y_pred_cls = tf.argmax(y_pred, 1)

    
  session = tf.Session(config=tf.ConfigProto(log_device_placement=True))

  session.run(tf.global_variables_initializer())
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer, labels=y_true)
  cost = tf.reduce_mean(cross_entropy)
  optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

  correct_prediction = tf.equal(y_pred_cls, y_true_cls)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  session.run(tf.global_variables_initializer())

if __name__ == '__main__':
    patch_size_1 = 100
    patch_size_2 = 100

    all_name = os.listdir('./Dataset_A/data')

    x_train_index, y_train = read_index("./Dataset_A/train")
    y_train = np.asarray(y_train, np.int32)

    x_train_patch, x_train_patch_index, image_index, y_train_patch, final_image_size \
        = get_data_in_patch(all_name, x_train_index, patch_size_1, patch_size_2, y_train)

    x_train_patch = np.asarray(x_train_patch)
    x_train_patch = x_train_patch.reshape((-1, 100, 100, 1))

    y_list = []
    for label in y_train_patch:
        if label == 1:
            y_list.append([0, 1])
        elif label == 0:
            y_list.append([1, 0])

    total_batches_tr = int(len(y_train_patch) / batch_size) + 1
    print(total_batches_tr)
    for epoch in range(total_epochs):
        acc_tr = 0
        for batch in range(total_batches_tr):
            train_batch = x_train_patch[batch * batch_size:(batch + 1) * batch_size]
            trainLabel_batch = y_list[batch * batch_size:(batch + 1) * batch_size]

            feed_tr = {x: train_batch, y_true: trainLabel_batch}

            session.run(optimizer, feed_dict=feed_tr)
            acc = session.run(accuracy, feed_dict=feed_tr)
            acc_tr += acc

            msg = "Training Epoch {0}  Batch {1} ----- Training Accuracy: {2:>6.1%}"
            print(msg.format(epoch + 1, batch + 1, acc))

        acc_tr_avg = acc_tr / total_batches_tr
        msg = "Training Epoch {0}---Training Accuracy: {1:>6.1%}"
        print(msg.format(epoch + 1, acc_tr_avg))

        saver = tf.train.Saver()
        name = "./CNN_" + str(epoch+1)
        saver.save(session, name)
    session.close()






