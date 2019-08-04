import math
import tensorflow as tf
from dataPreProcess import encodeWords
from dataPreProcess import createEncodedDataset
import os


def set_default_parameters():
    return tf.contrib.training.HParams(
        embedding_size=16,
        encoded_length=50,
        num_word_threshold=20,
        num_lstm_nodes=[32, 32],
        num_lstm_layers=2,
        num_fc_nodes=32,
        batch_size=100,
        learning_rate=0.001,
        clip_lstm_grads=1.0,
    )


def create_model(hps, vocab_size, classes_size):

    # 输入定义
    encoded_length = hps.encoded_length
    batch_size = hps.batch_size

    inputs = tf.placeholder(tf.int32, (batch_size, encoded_length))
    outputs = tf.placeholder(tf.int32, (batch_size, ))

    # for drop_out
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # record training step, un-trainable，保存当前训练到了那一步
    global_step = tf.Variable(
        tf.zeros([], tf.int64), name='global_step', trainable=False
    )

    # embedding layer
    # initialize embedding layer with uniform-distribution from -1 to +1
    embedding_init = tf.random_uniform_initializer(-1.0, 1.)
    with tf.variable_scope('embedding', initializer=embedding_init):
        embedding = tf.get_variable(
            'embedding',
            [vocab_size, hps.embedding_size],   # size of embedding matrix
            tf.float32
        )
        #
        embedded_inputs = tf.nn.embedding_lookup(embedding, inputs)

    # LSTM layers
    scale = 1.0/math.sqrt(hps.embedding_size + hps.num_lstm_nodes[-1])/3.0
    lstm_init = tf.random_uniform_initializer(-scale, scale)
    with tf.variable_scope('lstm', initializer=lstm_init):
        # store two layers
        cells = []
        for i in range(hps.num_lstm_layers):
            cell = tf.contrib.rnn.BasicLSTMCell(
                hps.num_lstm_nodes[i],
                state_is_tuple=True
            )
            cell = tf.contrib.rnn.DropoutWrapper(
                cell,
                output_keep_prob=keep_prob
            )
            cells.append(cell)
        # combine two LSTM layers: 第一个cell的输出为第二个cell的输入
        cell = tf.contrib.rnn.MultiRNNCell(cells)
        # init state
        initialize_state = cell.zero_state(batch_size, tf.float32)
        # input a sentence to cell, 此时就可以把句子输入到cell中

        # rnn_outputs: [batch_size, encoded_length, lstm_output[-1]]
        rnn_outputs, _ = tf.nn.dynamic_rnn(
            cell, embedded_inputs, initial_state=initialize_state)
        last = rnn_outputs[:, -1, :]

    # fc layer
    fc_init = tf.uniform_unit_scaling_initializer(factor=1.0)
    with tf.variable_scope('fc', initializer=fc_init):
        fc1 = tf.layers.dense(last, hps.num_fc_nodes, activation=tf.nn.relu, name='fc1')
        fc1_dropout = tf.contrib.layers.dropout(fc1, keep_prob)
        logits = tf.layers.dense(fc1_dropout, classes_size, name='fc2')

    # calculate loss function, y_pred, accuracy
    with tf.name_scope('metrics'):
        softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=outputs
        )
        loss = tf.reduce_mean(softmax_loss)
        y_pred = tf.argmax(tf.nn.softmax(logits), 1, output_type=tf.int32)
        correct_pred = tf.equal(outputs, y_pred)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.name_scope('train_op'):
        # get all trainable variables
        trainable_vars = tf.trainable_variables()
        # show all these trainable variables
        for var in trainable_vars:
            print('variable name: %s' % var)
            # tf.logging.info('variable name: %s' % var)
        # get all grads from loss with respect to all trainable variables
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(loss, trainable_vars), hps.clip_lstm_grads
        )

        # use AdamOptimizer
        optimizer = tf.train.AdamOptimizer(hps.learning_rate)

        # apply grads to all trainable_variables & train
        train_op = optimizer.apply_gradients(
            zip(grads, trainable_vars), global_step=global_step
        )

    return ((inputs, outputs, keep_prob),
            (loss, accuracy),
            (train_op, global_step))


""" Entry area """
seg_train_file = '../cnews_data/cnews.train.seg.txt'
seg_val_file = '../cnews_data/cnews.val.seg.txt'
seg_test_file = '../cnews_data/cnews.test.seg.txt'

vocab_file = '../cnews_data/cnews.vocab.txt'
category_file = '../cnews_data/cnews.category.txt'
output_folder = '../run_text_rnn'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

hps = set_default_parameters()

# create two instance for VocabDict & CategoryDict
vocab_instance = encodeWords.VocabDict(vocab_file, hps.num_word_threshold)
catego_instance = encodeWords.CategoryDict(category_file)


# execute this function:
placeholders, metrics, others = create_model(hps,
                                             vocab_instance.size(),
                                             catego_instance.size())
inputs, outputs, keep_prob = placeholders
loss, accuracy = metrics
train_op, global_step = others


init_op = tf.global_variables_initializer()
train_keep_prob = 0.8
test_keep_prob = 1.0  # no dropout for test data
num_train_steps = 1000

# encoded training data set:
train_dataset = createEncodedDataset.EncodedDataset(
        seg_train_file, vocab_instance, catego_instance, hps.encoded_length)

# encoded validation data set:
val_dataset = createEncodedDataset.EncodedDataset(
    seg_val_file, vocab_instance, catego_instance, hps.encoded_length)

# encoded test data set:
test_dataset = createEncodedDataset.EncodedDataset(
        seg_test_file, vocab_instance, catego_instance, hps.encoded_length)

with tf.Session() as sess:
    # init whole network
    sess.run(init_op)
    for i in range(num_train_steps):
        batch_inputs, batch_label = train_dataset.next_batch(hps.batch_size)
        # training: global_step+1 when sess.run() is called
        outputs_val = sess.run([loss, accuracy, train_op, global_step],
                               feed_dict={
                                   inputs: batch_inputs,
                                   outputs: batch_label,
                                   keep_prob: train_keep_prob
                               })
        # get three values from output_val
        loss_val, accuracy_val, _, global_step_val = outputs_val

        # print for every 100 times
        if global_step_val % 200 == 0:
            print("step: %5d, loss: %3.3f, accuracy: %3.5f" %
                  (global_step_val, loss_val, accuracy_val)
                  )
        # in every 1000 steps, do evaluation:
