import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

diabetes = pd.read_csv('/Users/Marko/Desktop/Udemy Tensorflow Course/NN Basics/pima-indians-diabetes.csv')

# preprocessing -----------------------------------------------------------------

columns_to_normalize = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps','Insulin', 'BMI', 'Pedigree']
diabetes[columns_to_normalize] = diabetes[columns_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# train test split -----------------------------------------------------------------

x_data = diabetes.drop(['Class', 'Group'], axis=1)
labels = diabetes['Class']
X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.4, random_state=101)
y_train = np.concatenate((y_train.values.reshape(-1,1), (1-y_train).values.reshape(-1,1)), axis=1)
y_test = np.concatenate((y_test.values.reshape(-1,1), (1-y_test).values.reshape(-1,1)), axis=1)

# model creation -----------------------------------------------------------------

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_nodes_hl4 = 500

n_features = 8
n_classes = 2

# 537 elements in X_train
batch_size = 33
num_batches = 16

# training hyperparameters
total_epochs = 3000
dropout_probability = 0.5
learn_rate = 0.002

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

def neural_network_model(data):

	initializer = tf.contrib.layers.xavier_initializer()
	
	hidden_1_layer = {'weights': tf.Variable(initializer([n_features, n_nodes_hl1])),
					'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_2_layer = {'weights': tf.Variable(initializer([n_nodes_hl1, n_nodes_hl2])),
					'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
	hidden_3_layer = {'weights': tf.Variable(initializer([n_nodes_hl2, n_nodes_hl3])),
					'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
	hidden_4_layer = {'weights': tf.Variable(initializer([n_nodes_hl3, n_nodes_hl4])),
					'biases': tf.Variable(tf.random_normal([n_nodes_hl4]))}
	output_layer = {'weights':tf.Variable(initializer([n_nodes_hl4, n_classes])),
					'biases':tf.Variable(tf.random_normal([n_classes]))}
	
	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)
	drop_out_1 = tf.nn.dropout(l1, keep_prob)
	
	l2 = tf.add(tf.matmul(drop_out_1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)
	drop_out_2 = tf.nn.dropout(l2, keep_prob)
	
	l3 = tf.add(tf.matmul(drop_out_2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)
	drop_out_3 = tf.nn.dropout(l3, keep_prob)

	l4 = tf.add(tf.matmul(drop_out_3, hidden_4_layer['weights']), hidden_4_layer['biases'])
	l4 = tf.nn.relu(l4)
	drop_out_4 = tf.nn.dropout(l4, keep_prob)
	# output shape -- [batch_size, 2]
	# example output = [[0.63, 0.37], 
	# 					[0.43, 0.57]]
	output = tf.add(tf.matmul(drop_out_4, output_layer['weights']), output_layer['biases'])

	return output
	
def train_neural_network(x):

	prediction = neural_network_model(x)
	softmax_test = tf.nn.softmax(prediction)

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost)

	per_epoch_correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(per_epoch_correct, tf.float32))
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		pred = []
		for epoch in range(total_epochs):
			acc = 0
			epoch_loss = 0
			i = 0
			while i < len(X_train)-9:
				start_index = i
				end_index = i + batch_size

				batch_x = np.array(X_train[start_index:end_index])
				batch_y = np.array(y_train[start_index:end_index])

				_ , c, acc, pred = sess.run([optimizer, cost, accuracy, softmax_test], feed_dict={x: batch_x, y:batch_y, keep_prob:dropout_probability})
				epoch_loss += c
				i += batch_size
			print(pred[0:10])
			print('Epoch {} completed out of {} loss: {:.9f} accuracy: {:.9f}'.format(epoch+1, total_epochs, epoch_loss, acc))

		# get accuracy

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		final_accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
		print('Accuracy:', final_accuracy.eval({x:X_test, y:y_test, keep_prob:1.0}))  

		
		
if __name__ == "__main__":
	train_neural_network(x)











