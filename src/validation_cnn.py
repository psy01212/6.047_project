import random
from os.path import join
import gzip
import math
import tensorflow as tf

random.seed(42)

## Getting all the cell types that have 7 features (everything excluding DNase)
## predicting H3K9ac based on other marks (excluding DNase)

chrom = "chr21"
file_end = ".pval.signal.bedGraph.wig.gz"
datadir = join("..", "data", "all_data", "EXAMPLE", "CONVERTEDDATADIR")

# split: 30/9/10
train_cells = []
validation_cells = []
test_cells = []
all_marks = ["H3K4me1", "H3K4me3", "H3K36me3", "H3K27me3", "H3K9me3", "H3K27ac", "H3K9ac"]
x_marks = ["H3K4me1", "H3K4me3", "H3K36me3", "H3K27me3", "H3K9me3", "H3K27ac"]
y_mark = "H3K9ac"

with open(join("r2_scores", "H3K9ac_H3K27ac.txt")) as f:
    # skip first 5 lines
    for i in range(5):
        f.readline()
    counter = 0
    for i in range(30):
        train_cells.append(f.readline()[:4])
    for i in range(9):
        validation_cells.append(f.readline()[:4])
    for i in range(10):
        test_cells.append(f.readline()[:4])

def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride=1):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

def max_pool(x, stride=2, filter_size=2):
    return tf.nn.max_pool(x, ksize=[1, filter_size, 1, 1],
                        strides=[1, stride, 1, 1], padding='SAME')

def build_two_fc_layers(x_inp, Ws, bs, keep_prob=1):
    h_fc1 = tf.nn.relu(tf.matmul(x_inp, Ws[0]) + bs[0])
    h_dropout = tf.nn.dropout(h_fc1, keep_prob)
    return tf.matmul(h_dropout, Ws[1]) + bs[1]

def l1_loss(x):
    return tf.reduce_sum(tf.abs(x))

## shape of the neural network and parameters
x_dim = 101
mid_index = 50
y_dim = 1
num_channels = 6
num_classes = 1

conv1_filter_size = 5
conv1_depth = 8

conv2_filter_size = 5
conv2_depth = 16

batch_size = 50
learning_rate = 0.0001
fc_num_hidden = 64
keep_prob = 0.7
##reg_lambda = 0.0001

## convolutional layers
tf.reset_default_graph()

train_x = tf.placeholder(tf.float32, shape=(None, x_dim, y_dim, num_channels))
train_y = tf.placeholder(tf.float32, shape=(None, num_classes))

##x_image = tf.reshape(train_x, [-1, x_dim, y_dim, num_channels])

conv1_filter = weight_variable([conv1_filter_size, 1, num_channels, conv1_depth])
conv1_bias = bias_variable([conv1_depth])
h_conv1 = tf.add(conv2d(train_x, conv1_filter), conv1_bias)
h_pool1 = max_pool(h_conv1)

conv2_filter = weight_variable([conv2_filter_size, 1, conv1_depth, conv2_depth])
conv2_bias = bias_variable([conv2_depth])
h_conv2 = tf.add(conv2d(h_pool1, conv2_filter), conv2_bias)
h_pool2 = max_pool(h_conv2)

## fully connected layers

conv2_feat_map_x = 26 # 101 / 2 / 2
conv2_feat_map_y = 1

W_fc1 = weight_variable([conv2_feat_map_x * conv2_feat_map_y * conv2_depth, fc_num_hidden])
b_fc1 = bias_variable([fc_num_hidden])

h_pool2_flat = tf.reshape(h_pool2, [-1, conv2_feat_map_x * conv2_feat_map_y * conv2_depth])

W_fc2 = weight_variable([fc_num_hidden, num_classes])
b_fc2 = bias_variable([num_classes])

y_conv = build_two_fc_layers(h_pool2_flat, [W_fc1, W_fc2], [b_fc1, b_fc2], keep_prob)

## loss is mse + l2 loss
mse = tf.losses.mean_squared_error(y_conv, train_y)
loss = mse
##loss = loss + reg_lambda * (tf.nn.l2_loss(conv1_filter) + tf.nn.l2_loss(conv2_filter) + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2))

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

##### starting training
print 'Initializing variables...'
sess = tf.Session()
init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=20)

sess.run(init)

saver.restore(sess, join("saved_models", "model-0"))

print "Caluating validation metrics"
num_samples_epoch = 0
epoch_loss = 0.
epoch_mse = 0.
epoch_num_top_percent = 0
epoch_num_correct_top_percent = 0
for cell in validation_cells:
    ## get all xs
    all_xs = []
    all_ys = []
    all_files = []
    for mark_i in range(len(all_marks)):
        filename = chrom + "_" + cell + "-" + all_marks[mark_i] + file_end
        all_files.append(gzip.open(join(datadir, filename)))
        all_files[mark_i].readline()
        all_files[mark_i].readline()

    ## need to pad the x features with 50 in the front/back, since we need a window of 101 to predict middle feature (y)
    is_firstline = True
    for line in all_files[0]:
        all_lines = [float(line.decode("utf-8"))]
        for f in all_files[1:]:
            all_lines.append(float(f.readline().decode("utf-8")))
        all_xs.append([[val for val in all_lines[:-1]]])
        all_ys.append([all_lines[-1]])
        if is_firstline:
            for i in range(50):
                all_xs.append([[val for val in all_lines[:-1]]])
            is_firstline = False
        # else do nothing because it's a duplicate
    ## pad the back
    for i in range(50):
        all_xs.append([[val for val in all_lines[:-1]]])
    for f in all_files:
        f.close()
    print "Done reading files for cell " + cell
    print len(all_ys)

    cell_loss = 0.
    cell_mse = 0.
    num_samples_cell = 0
    all_y_conv = []

    ## calculating y's in batches of 50
    ## all_xs is 100 bases longer than all_ys because of appending 50 to each ends
    num_batches_cell = len(all_ys)/batch_size
    num_leftover = len(all_ys) % batch_size
    for i in range(num_batches_cell):
        batch_x = []
        batch_y = []
        for j in range(batch_size):
            start_index = batch_size * i + j
            batch_x.append(all_xs[start_index:start_index+x_dim])
            batch_y.append(all_ys[start_index])
        feed_dict = {train_x: batch_x, train_y: batch_y}
        batch_y_conv, batch_loss, batch_mse = sess.run([y_conv, loss, mse], feed_dict=feed_dict)
        if not math.isnan(batch_loss) and not math.isnan(batch_mse):
            cell_loss += batch_loss
            cell_mse += batch_mse
            num_samples_cell += 1
        for single_y_conv in batch_y_conv:
            all_y_conv.append(single_y_conv)
    ## also append the remaining (the last samples that didn't fit in the batch of 50)
    batch_x = []
    batch_y = []
    for j in range(num_leftover):
        start_index = batch_size * num_batches_cell + j
        batch_x.append(all_xs[start_index:start_index+x_dim])
        batch_y.append(all_ys[start_index])
    feed_dict = {train_x: batch_x, train_y: batch_y}
    batch_y_conv, batch_loss, batch_mse = sess.run([y_conv, loss, mse], feed_dict=feed_dict)
    if not math.isnan(batch_loss) and not math.isnan(batch_mse):
        cell_loss += batch_loss
        cell_mse += batch_mse
        num_samples_cell += 1
    for single_y_conv in batch_y_conv:
        all_y_conv.append(single_y_conv)

    ### calculate metric
    num_top_percent = len(all_ys) / 100
    y_top_indices = sorted(range(len(all_ys)), key=lambda i: all_ys[i][0], reverse=True)[:num_top_percent]

    y_top_indices_set = set()
    for i in y_top_indices:
        y_top_indices_set.add(i)
        
    y_top_indices = sorted(range(len(all_y_conv)), key=lambda i: all_y_conv[i][0], reverse=True)[:num_top_percent]
    num_correct_top_percent = 0
    for i in y_top_indices:
        if i in y_top_indices_set:
            num_correct_top_percent += 1

    num_samples_epoch += num_samples_cell
    epoch_loss += cell_loss
    epoch_mse += cell_mse
    epoch_num_top_percent += num_top_percent
    epoch_num_correct_top_percent += num_correct_top_percent
    print "Samples in cell: " + str(num_samples_cell)
    print "Average cell loss: " + str(cell_loss / num_samples_cell)
    print "Average cell mse: " + str(cell_mse / num_samples_cell)
    print "Proportion of top 1% correctly guessed: " + str(float(num_correct_top_percent) / num_top_percent)
print "EPOCH DONE"
print "Average epoch loss: " + str(epoch_loss / num_samples_epoch)
print "Average epoch mse: " + str(epoch_mse / num_samples_epoch)
print "Proportion of top 1% correctly guessed: " + str(float(epoch_num_correct_top_percent) / epoch_num_top_percent)
