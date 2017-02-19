"""
Notes: This script is a basic implementation of RNN (LSTM) with variable
length input. The script takes same sized input in all cases but runs only
only till specific timesteps required.

Many things are not clear yet. How to pre-allocate chunk of GPU memory for
computation? Does creation of tensor placeholders do that?

It turns out that different early_stop value leads to different run-time for
the same input. Hence, early_stop passed to rnn can help us get better
running-time. The output sahpe for different early_stop is same, hence,
it is not clear what is the value of output for the timesteps which was not
computed.
"""
import tensorflow as tf
from tensorflow.contrib.rnn import static_rnn
from tensorflow.contrib.rnn import LSTMCell
from time import time
import numpy as np

initializer = tf.random_uniform_initializer(-1, 1)
seq_input = tf.placeholder(shape=[200, 10, 2048], dtype=tf.float32)
inputs = [ts[0] for ts in tf.split(seq_input, 200, axis=0)]
early_stop = tf.placeholder(dtype=tf.int32)
cell = LSTMCell(2048, initializer=initializer)
initial_state = cell.zero_state(10, dtype=tf.float32)
outputs, states = static_rnn(
    cell,
    inputs,
    initial_state=initial_state,
    sequence_length=early_stop
    )


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()


input_values = np.random.uniform(size=(200, 10, 2048)).astype('float32')
start_time = time()
for i in range(100):
    output = sess.run(outputs, feed_dict={
        seq_input: input_values,
        early_stop: 10})

print len(output)
tot_time = time() - start_time
print("Per iteration time for 10 timesteps is {}".format(tot_time/100.))

# input_values = [np.random.uniform(size=(10, 2048)) for i in range(100)]
start_time = time()
for i in range(100):
    output = sess.run(outputs, feed_dict={
        seq_input: input_values,
        early_stop: 100})

print len(output)

tot_time = time() - start_time
print("Per iteration time for 100 timesteps is {}".format(tot_time/100.))

# input_values = [np.random.uniform(size=(10, 2048)) for i in range(200)]
start_time = time()
for i in range(100):
    output = sess.run(outputs, feed_dict={
        seq_input: input_values,
        early_stop: 200})

print len(output)

tot_time = time() - start_time
print("Per iteration time for 200 timesteps is {}".format(tot_time/100.))
