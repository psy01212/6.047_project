import random
from os.path import join
import gzip
import tensorflow-gpu as tf

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

learning_rate = 0.01
fc_num_hidden = 64
keep_prob = 0.6
reg_lambda = 0.0001

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
loss = mse + reg_lambda * (tf.nn.l2_loss(conv1_filter) + tf.nn.l2_loss(conv2_filter) + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2))

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

##### starting training
print 'Initializing variables...'
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print "Starting training..."
# every step is a full round of 30 test cell types + 9 validation cell types
for epoch in range(10):
    epoch_loss = 0
    epoch_mse = 0
    for cell in test_cells:
        cell_loss = 0
        cell_mse = 0
        ## get all xs
        all_xs = []
        for mark_i in range(len(x_marks)):
            filename = chrom + "_" + cell + "-" + x_marks[mark_i] + file_end
            if mark_i == 0:
                with gzip.open(join(datadir, filename)) as f:
                    f.readline()
                    f.readline()
                    for line in f:
                        all_xs.append([[float(line.decode("utf-8"))]])
            else:
                with gzip.open(join(datadir, filename)) as f:
                    f.readline()
                    f.readline()
                    line_num = 0
                    for line in f:
                        all_xs[line_num][0].append(float(line.decode("utf-8")))
                        line_num += 1
        all_ys = []
        filename = chrom + "_" + cell + "-" + y_mark + file_end
        with gzip.open(join(datadir, filename)) as f:
            f.readline()
            f.readline()
            for line in f:
                all_ys.append([float(line.decode("utf-8"))])
        print "Done reading files for cell " + cell
        for i in range(len(all_ys) - x_dim):
            if i%(len(all_ys)/100) == 0:
                print(i/(len(all_ys)/100))
            batch_x = [all_xs[i:i+x_dim]]
            batch_y = [all_ys[i+mid_index]]
            
            feed_dict = {train_x: batch_x, train_y: batch_y}
            _, batch_loss, batch_mse = sess.run([train_step, loss, mse],
                                                      feed_dict=feed_dict)
            cell_loss += batch_loss
            cell_mse += batch_mse
        epoch_loss += cell_loss
        epoch_mse += cell_mse
        print cell
        print "Cell loss: " + str(cell_loss)
        print "Cell mse: " + str(cell_mse)
    print "EPOCH DONE"
    print "Epoch loss: " + str(epoch_loss)
    print "Epoch mse: " + str(epoch_mse)
