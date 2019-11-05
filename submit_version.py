import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
import matplotlib.pyplot as plt
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# hyper parameters
kernel_size = 5
img_w = 200
img_h = 200
feature_map_num = [1, 8, 16, 32]
fc_size = 64
output_size = 3
batch_size = 128
start_learning_rate = 0.01
decay_rate = 0.96
epochs_num = 100
batch_num = 1000
decay_steps = batch_num

def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
        (rr >= 0) &
        (rr < img.shape[0]) &
        (cc >= 0) &
        (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]

def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img

def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
        shape0.intersection(shape1).area /
        shape0.union(shape1).area
    )

def test():  
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    saver = tf.train.import_meta_graph('model/model.ckpt.meta')
    graph = tf.get_default_graph()
    saver.restore(sess,tf.train.latest_checkpoint('model/'))

    x = graph.get_tensor_by_name('Input/img:0')
    train_mode = graph.get_tensor_by_name('Input/train_mode:0')
    prediction = graph.get_tensor_by_name('prediction:0')

    train_flag = tf.Variable(False, tf.bool)
    results = []
    for i in range(1000):
        params, img = noisy_circle(200, 50, 2)
        x_batch = []
        x_batch.append(img)
        detected = sess.run([prediction], 
                            feed_dict={x: x_batch, 
                                       train_mode: train_flag})
        detected = np.squeeze(detected)
        results.append(iou(params, detected))
    results = np.array(results)
    print((results > 0.7).mean())

def batch_norm(x, n_out, train_mode):
  with tf.variable_scope('BN'):
    beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                  name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                  name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    
    mean, var = tf.cond(train_mode,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolutional_neural_network(x, train_mode):
  with tf.variable_scope('Weight'):
    weights = {'W_conv1':tf.Variable(tf.random_normal([kernel_size, kernel_size, 1, feature_map_num[1]]), name = 'W_conv1'),
                'W_conv2':tf.Variable(tf.random_normal([kernel_size, kernel_size, feature_map_num[1], feature_map_num[2]]), name = 'W_conv2'),
                'W_conv3':tf.Variable(tf.random_normal([kernel_size, kernel_size, feature_map_num[2], feature_map_num[3]]), name = 'W_conv3'),
                'W_conv4':tf.Variable(tf.random_normal([1, 1, feature_map_num[3], 1]), name = 'W_conv4'),
                'W_fc':tf.Variable(tf.random_normal([int(img_w/4) * int(img_h/4), fc_size]), name = 'W_fc'),
                'W_out':tf.Variable(tf.random_normal([fc_size, output_size]), name = 'W_out')}

    biases = {'b_conv1':tf.Variable(tf.random_normal([feature_map_num[1]]), name = 'b_conv1'),
              'b_conv2':tf.Variable(tf.random_normal([feature_map_num[2]]), name = 'b_conv2'),
              'b_conv3':tf.Variable(tf.random_normal([feature_map_num[3]]), name = 'b_conv3'),
              'b_conv4':tf.Variable(tf.random_normal([1]), name = 'b_conv4'),
              'b_fc':tf.Variable(tf.random_normal([fc_size]), name = 'b_fc'),
              'b_out':tf.Variable(tf.random_normal([output_size]), name = 'b_out')}

  x = tf.reshape(x, shape=[-1, img_w, img_h, 1])
  conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
  conv1 = batch_norm(conv1, feature_map_num[1], train_mode)

  conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
  conv2 = batch_norm(conv2, feature_map_num[2], train_mode)
  conv2 = maxpool2d(conv2)

  conv3 = tf.nn.relu(conv2d(conv2, weights['W_conv3']) + biases['b_conv3'])
  conv3 = batch_norm(conv3, feature_map_num[3], train_mode)
  conv3 = maxpool2d(conv3)
  
  conv4 = tf.nn.relu(conv2d(conv3, weights['W_conv4']) + biases['b_conv4'])
  conv4 = batch_norm(conv4, 1, train_mode)

  fc = tf.reshape(conv4,[-1, int(img_w/4) * int(img_h/4)])
  fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
  prediction = tf.add(tf.matmul(fc, weights['W_out']), biases['b_out'], name = 'prediction')
  return prediction

def train_neural_network():
  with tf.variable_scope('Input'):
    x = tf.placeholder(tf.float32, shape=[None, img_w, img_h], name = "img")
    y = tf.placeholder(tf.float32, shape = [None, output_size], name = "parameters")
    global_step = tf.Variable(0, trainable=False, name = 'global_step')
    train_mode = tf.Variable(True, tf.bool, name = 'train_mode')

  prediction = convolutional_neural_network(x, train_mode)
  cost = tf.math.reduce_sum(tf.math.square(prediction - y)) / batch_size
  learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,
                                          decay_steps, decay_rate, staircase=True)
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)

  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    sess.run(tf.initialize_all_variables())
    
    vars_to_save_dict = {}
    scopelist = ['BN', 'Weight']
    for s in scopelist:
      vars_to_save = [v for v in tf.global_variables() if s in v.name and 'Adam' not in v.name]
      for v in vars_to_save:
        vars_to_save_dict[v.name[:-2]] = v
    
    f = open('model/output.txt', 'w')
    for epoch in range(epochs_num):
        for i in range(batch_num):
            x_batch = []
            y_batch = []
            for _ in range(batch_size):
                params, img = noisy_circle(200, 50, 2)
                y_batch.append(params), x_batch.append(img)
            _, loss, pre, label, lr = sess.run([optimizer, cost, prediction, y, learning_rate], 
                                               feed_dict={x: x_batch, y: y_batch})
            print('epoch: ', epoch, 
                  '  batch: ', i, 
                  '  loss: ', loss, 
                  '  prediction: ', pre[0], 
                  '  label: ', label[0], 
                  ' learning_rate: ', lr)  
            f.write('epoch: ' + str(epoch) + ',  batch: ' + str(i) + ',  loss: ' + str(loss))
            f.write('\n')
            
        saver = tf.train.Saver(vars_to_save_dict)
        save_path = saver.save(sess, "model/model.ckpt")
    f.close()

if __name__ == '__main__':
  train_neural_network()
  test()

