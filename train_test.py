"""
Created on Tue Dec  4 19:35:30 2018

@author:Nagendra athreya

Ref: https://github.com/haotianteng/Chiron
"""
#import matplotlib.pyplot as plt
#import numpy as np
import tensorflow as tf
import time
import read_signal_label
from cnn import getcnnfeature
#from cnn import getcnnlogit
from rnn-att import rnn_layers, rnn_layers_one_direction
#from utils.attention import attention_loss

LR_BOUNDARY = [0.66,0.83]
LR_DECAY = [1e-1,1e-2]

class Flags():
    def __init__(self):
        self.home_dir = '/home/conjugacy/Downloads/project'
        self.data_dir = '/home/conjugacy/Downloads/project/train/'
        self.valid_dir = '/home/athreya/Dropbox/CS598-JP/project/valid/'
        self.log_dir = '/home/conjugacy/Downloads/project/log/'
        self.h5py_file_path = '/home/conjugacy/Downloads/project/h5py_file_path'
        self.sequence_len = 400
        self.batch_size = 200
        self.step_rate = 1e-3
        self.max_steps = 10000
        self.model_name = 'crnn_new7_ctc_bak'
        self.max_segments_number = None
        self.MAXLEN = 1e4

FLAGS = Flags()

def inference(x, seq_length, training):
    cnn_feature = getcnnfeature(x, training=training)
    feashape = cnn_feature.get_shape().as_list()
    ratio = FLAGS.sequence_len/feashape[1]
    logits = rnn_layers(cnn_feature, seq_length, training)
    return logits, ratio


def loss(logits, seq_len, label):
    loss = tf.reduce_mean(tf.nn.ctc_loss(label, logits, seq_len, ctc_merge_repeated=True, time_major=False, ignore_longer_outputs_than_inputs=True))
    tf.add_to_collection('losses',loss)
    """Note here ctc_loss will perform softmax, so no need to softmax the logits."""
    tf.summary.scalar('loss', loss)
    return tf.add_n(tf.get_collection('losses'),name = 'total_loss')


def train_opt():
    boundaries = [int(FLAGS.max_steps*LR_BOUNDARY[0]), int(FLAGS.max_steps*LR_BOUNDARY[1])]
    values = [FLAGS.step_rate * decay for decay in [1,LR_DECAY[0],LR_DECAY[1]]]
    learning_rate = tf.train.piecewise_constant(global_step,boundaries,values)
    opt = tf.train.AdamOptimizer(learning_rate)
    return opt


def prediction(logits, seq_length, label, top_paths=1):
    logits = tf.transpose(logits, perm=[1, 0, 2])
    """ctc_beam_search_decoder require input shape [max_time,batch_size,num_classes]"""
    predict = tf.nn.ctc_beam_search_decoder(logits, seq_length, merge_repeated=False, top_paths=top_paths)[0]
    edit_d = list()
    for i in range(top_paths):
        tmp_d = tf.edit_distance(tf.to_int32(predict[i]), label, normalize=True)
        edit_d.append(tmp_d)
    tf.stack(edit_d, axis=0)
    d_min = tf.reduce_min(edit_d, axis=0)
    error = tf.reduce_mean(d_min, axis=0)
    tf.summary.scalar('Error_rate', error)
    return error

"""Copy the train function here"""
#train_ds, valid_ds = read_signal_label.read_raw_data_sets(FLAGS.data_dir, FLAGS.sequence_len, valid_reads_num=100)


training = tf.placeholder(tf.bool)
global_step = tf.get_variable('global_step', trainable=False, shape=(),
                              dtype=tf.int32,
                              initializer=tf.zeros_initializer())
x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.sequence_len])
seq_length = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
y_indexs = tf.placeholder(tf.int64)
y_values = tf.placeholder(tf.int32)
y_shape = tf.placeholder(tf.int64)
y = tf.SparseTensor(y_indexs, y_values, y_shape)

logits, ratio = inference(x, seq_length, training)

ctc_loss = loss(logits, seq_length, y)

opt = train_opt()
#step = opt.minimize(ctc_loss,global_step = global_step)
grads, variables = zip(*opt.compute_gradients(ctc_loss))
grads, _ = tf.clip_by_global_norm(grads, 50.0) # gradient clipping
grads_and_vars = list(zip(grads, variables))
step = opt.apply_gradients(grads_and_vars)

error = prediction(logits, seq_length, y)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
summary = tf.summary.merge_all()

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
sess = tf.Session(config=config)
sess.run(init)
summary_writer = tf.summary.FileWriter(FLAGS.log_dir + FLAGS.model_name + '/summary/', sess.graph)

#_ = tf.train.start_queue_runners(sess=sess)
start = time.time()
#saver.restore(sess,FLAGS.log_dir+FLAGS.model_name+'/model.ckpt-9070')
#saver.restore(sess, tf.train.latest_checkpoint(FLAGS.log_dir + FLAGS.model_name))
#batch_x, seq_len, batch_y = train_ds.next_batch(FLAGS.batch_size, shuffle=False)
#indxs, values, shape = batch_y
#feed_dict = {x: batch_x, seq_length: seq_len, y_indexs: indxs, y_values: values, y_shape: shape, training: False}
#loss_val = sess.run([ctc_loss], feed_dict=feed_dict)
# valid_x,valid_len,valid_y = valid_ds.next_batch(FLAGS.batch_size)
# feed_dict = {x:batch_x,seq_length:seq_len,y_indexs:indxs,y_values:values,y_shape:shape,training:False}
#error_val = sess.run(error, feed_dict=feed_dict)

#train_ds = read_signal_label.read_raw_data_sets(FLAGS.data_dir, h5py_file_path=None, seq_length=FLAGS.sequence_len, k_mer=1, max_segments_num=FLAGS.max_segments_number)
train_ds = read_signal_label.read_cache_dataset(FLAGS.h5py_file_path)
valid_ds = train_ds

for i in range(FLAGS.max_steps):
    batch_x, seq_len, batch_y = train_ds.next_batch(FLAGS.batch_size)
    indxs, values, shape = batch_y
    feed_dict = {x: batch_x, seq_length: seq_len / ratio, y_indexs: indxs,
                 y_values: values, y_shape: shape,
                 training: True}
    loss_val, _ = sess.run([ctc_loss, step], feed_dict=feed_dict)
    if i % 10 == 0:
        global_step_val = tf.train.global_step(sess, global_step)
        valid_x, valid_len, valid_y = valid_ds.next_batch(FLAGS.batch_size)
        indxs, values, shape = valid_y
        feed_dict = {x: valid_x, seq_length: valid_len / ratio,
                     y_indexs: indxs, y_values: values, y_shape: shape,
                     training: True}
        error_val = sess.run(error, feed_dict=feed_dict)
        end = time.time()
        print(
        "Step %d/%d Epoch %d, batch number %d, train_loss: %5.3f validate_edit_distance: %5.3f Elapsed Time/step: %5.3f" \
        % (i, FLAGS.max_steps, train_ds.epochs_completed,
           train_ds.index_in_epoch, loss_val, error_val,
           (end - start) / (i + 1)))
        saver.save(sess, FLAGS.log_dir + FLAGS.model_name + '/model.ckpt',
                   global_step=global_step_val)
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, global_step=global_step_val)
        summary_writer.flush()
global_step_val = tf.train.global_step(sess, global_step)
print("Model %s saved." % (FLAGS.log_dir + FLAGS.model_name))
print("Reads number %d" % (train_ds.reads_n))
saver.save(sess, FLAGS.log_dir + FLAGS.model_name + '/final.ckpt',
           global_step=global_step_val)

""""""
'''
"""Conduct test"""
batch_x,seq_len,batch_y = train_ds.next_batch(FLAGS.batch_size)
indxs, values, shape = batch_y
feed_dict = {x:batch_x,seq_length:seq_len,y_indexs:indxs,y_values:values,y_shape:shape,training:True}
predict = tf.nn.ctc_beam_search_decoder(tf.transpose(logits, perm=[1, 0, 2]), seq_length, merge_repeated=False,
                                        top_paths=5)
predict_val = sess.run(predict, feed_dict=feed_dict)
predict_val_top5 = predict_val[0]
index_val = sess.run(y_indexs, feed_dict=feed_dict)
y_val_eval = sess.run(y_values, feed_dict=feed_dict)
index_val_bat = index_val[:, 0]
predict_read = list()
true_read = list()
for i in range(len(predict_val_top5)):
    predict_val = predict_val_top5[i]
    unique, len_counts = np.unique(index_val_bat, return_counts=True)
    unique, pre_counts = np.unique(predict_val.indices[:, 0], return_counts=True)
    pos_predict = 0
    pos_true = 0
    predict_read_temp = list()
    true_read_temp = list()
    for indx, counts in enumerate(len_counts):
        predict_read_temp.append(predict_val.values[pos_predict:pos_predict + pre_counts[indx]])
        pos_predict += pre_counts[indx]
        true_read_temp.append(y_val_eval[pos_true:pos_true + counts])
        pos_true += counts
    true_read.append(true_read_temp)
    predict_read.append(predict_read_temp)
""""""

"""logits plot"""
inspect_indx = 0
print(predict_read[0][inspect_indx].tolist())
print(true_read[0][inspect_indx].tolist())
logits_val = sess.run(logits, feed_dict=feed_dict)
logits_val = logits_val[10]
A_logits = logits_val[:, 0]
G_logits = logits_val[:, 1]
C_logits = logits_val[:, 2]
T_logits = logits_val[:, 3]
b_logits = logits_val[:, 4]
x_val = sess.run(x, feed_dict=feed_dict)
x_val = x_val[10]
plt.plot(x_val)
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(x_val)
axarr[0].set_title('signal')
axarr[1].plot(A_logits, color='r')
axarr[1].plot(G_logits, color='g')
axarr[1].plot(C_logits, color='b')
axarr[1].plot(T_logits, color='yellow')
axarr[1].plot(b_logits)
axarr[1].set_title('Base prediction')
'''
