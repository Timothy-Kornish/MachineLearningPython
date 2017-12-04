import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


#-------------------------------------------------------------------------------
#  Loading in MNist Data Set from tensor flow, dataset of hand-written numbers
#-------------------------------------------------------------------------------

mnist = input_data.read_data_sets('/tmp/data', one_hot = True)

print("\n-------------------------------------------------------------------\n")
print("mnist type: ", type(mnist))
print("\n-------------------------------------------------------------------\n")
print("mnist shape: ", mnist.train.images.shape)
print("\n-------------------------------------------------------------------\n")
sample = mnist.train.images[2].reshape(28, 28) # respahing array into square grid
print(sample, ' = 28 x 28 array')

#-------------------------------------------------------------------------------
#           Plotting mnist numbers
#-------------------------------------------------------------------------------


plt.imshow(sample, cmap = 'Greys')
plt.show()

t = mnist.train.next_batch(1)

xSamp, ySamp = t # tuple unpacking
plt.imshow(xSamp.reshape(28, 28), cmap = 'Greys')
plt.show()


#-------------------------------------------------------------------------------
#           Learning rate and Cost function
#-------------------------------------------------------------------------------

learning_rate = 0.001
training_epochs = 15 # number of training cycles
batch_size = 100

n_classes = 10
n_samples = mnist.train.num_examples
n_input = 784

#-------------------------------------------------------------------------------
#    Hidden Layers, more layers can make it more accurate at cost of runtime
#-------------------------------------------------------------------------------

n_hidden_1 = 256 # hidden perception layer, number of neurons
n_hidden_2 = 256 # hidden perception layer, number of neurons

#-------------------------------------------------------------------------------
#    Function for multi-layer perceptron
#-------------------------------------------------------------------------------

def multilayer_perceptron(x, weights, biases):
    """
    x: Placeholder for data input
    weights: Dict of weights
    biases: dict of bias values
    """

    # First Hidden Layer with RELU Activation
    # X * W + B
    layer_1 = tf.add(tf.matmul(x, weights['h1']) , biases['b1'])
    # RELU func(X * W + B) -> f(x) = max(0, x)
    layer_1 = tf.nn.relu(layer_1)

    # Second Hidden Layer
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2) # nn for neural netword

    # Last Output Layer
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    return out_layer

#-------------------------------------------------------------------------------
#           Setting weights and biases
#-------------------------------------------------------------------------------

weights = {
    'h1' : tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out' : tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
    'b2' : tf.Variable(tf.random_normal([n_hidden_2])),
    'out' : tf.Variable(tf.random_normal([n_classes]))
}
x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float', [None, n_classes])

#-------------------------------------------------------------------------------
#           predicting model
#-------------------------------------------------------------------------------

pred = multilayer_perceptron(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

#-------------------------------------------------------------------------------
#           Training the model
#-------------------------------------------------------------------------------

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

print("\n-------------------------------------------------------------------\n")
for epoch in range(training_epochs): # 15 loops
    # Cost
    avg_cost = 0.0
    total_batch = int(n_samples/batch_size)

    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _,c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
        avg_cost += c/total_batch

    print("Epoch: {} cost {:.4f}".format(epoch+1, avg_cost))
print("Model has completed {} Epochs of training".format(training_epochs))
print("\n-------------------------------------------------------------------\n")

#-------------------------------------------------------------------------------
#           Model Evaluations
#-------------------------------------------------------------------------------

correct_predictions = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
print(correct_predictions[0])
print("\n-------------------------------------------------------------------\n")

correct_predictions = tf.cast(correct_predictions, 'float')
print(correct_predictions[0])
print("\n-------------------------------------------------------------------\n")

accuracy = tf.reduce_mean(correct_predictions)

print("Accuracy: ", accuracy.eval({x : mnist.test.images, y : mnist.test.labels }))
print("\n-------------------------------------------------------------------\n")
