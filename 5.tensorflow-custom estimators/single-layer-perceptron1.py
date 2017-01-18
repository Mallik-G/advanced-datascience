import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from sklearn import model_selection
import numpy as np
import os


# Set logging level to info to see detailed log output
tf.logging.set_verbosity(tf.logging.INFO)

os.chdir("/home/algo")

sample = learn.datasets.base.load_csv_with_header(
      filename="train.csv",
      target_dtype=np.int,
      features_dtype=np.float32, target_column=-1)

X = sample.data
y = sample.target

# Divide the input data into train and validation
X_train,X_validation,y_train,y_validation = model_selection.train_test_split(X,y, test_size=0.2, random_state=100)
type(X_train)

o = np.array([0,1,2,1,2,1,0])
tmp  = tf.one_hot(o, 3, 1, 0)

x = np.array([10,100,0,-10,-100], dtype=np.float32)
sg = tf.sigmoid(x)

y = np.array([0.6,0.7,0.9,0.9], dtype=np.float32)
sm = tf.nn.softmax(y)

features = np.array(
                    [[1,2],[3,4]], dtype=np.float32)
nnout1 = layers.fully_connected(
    inputs=features,
    weights_initializer=tf.constant_initializer([1.0]),
    biases_initializer=tf.constant_initializer([1.0]),
    num_outputs=1,
    activation_fn=None)

nnout2 = layers.fully_connected(
    inputs=features,
    weights_initializer=tf.constant_initializer([1.0]),
    biases_initializer=tf.constant_initializer([1.0]),
    num_outputs=1,
    activation_fn=tf.sigmoid)

nnout3 = layers.fully_connected(
    inputs=features,
    weights_initializer=tf.constant_initializer([1.0]),
    biases_initializer=tf.constant_initializer([1.0]),
    num_outputs=2,
    activation_fn=None)

nnout4 = layers.fully_connected(
    inputs=features,
    weights_initializer=tf.constant_initializer([[1.0,1.0],[2.0,2.0]]),
    biases_initializer=tf.constant_initializer([1.0,2.0]),
    num_outputs=2,
    activation_fn=None)

nnout5 = layers.fully_connected(
    inputs=features,
    weights_initializer=tf.constant_initializer([[1.0,1.0],[2.0,2.0]]),
    biases_initializer=tf.constant_initializer([1.0,2.0]),
    num_outputs=2,
    activation_fn=tf.sigmoid)

nnout6 = layers.fully_connected(
    inputs=features,
    weights_initializer=tf.constant_initializer([[1.0,1.0],[2.0,2.0]]),
    biases_initializer=tf.constant_initializer([1.0,2.0]),
    num_outputs=2,
    activation_fn=tf.nn.softmax)

session = tf.Session()
session.run(tf.initialize_all_variables())
session.run(nnout6)

# creating custom estimator
def model_function(features, targets):

  targets = tf.one_hot(targets, 2, 1, 0)

  # Connect the output layer to second hidden layer (no activation fn)
  outputs = layers.fully_connected(inputs=features,
                                                 num_outputs=2,
                                                 activation_fn=tf.sigmoid)

  outputs_dict = {"labels": outputs}
  
  
  # Calculate loss using mean squared error
  loss = losses.mean_squared_error(outputs, targets)

  # Create training operation
  optimizer = layers.optimize_loss(
      loss=loss,
      global_step=tf.contrib.framework.get_global_step(),
      learning_rate=0.001,
      optimizer="SGD")

  return outputs_dict, loss, optimizer 


#create custom neural network model
nn = learn.Estimator(model_fn=model_function, params=model_params)

#build the model
nn.fit(x=X_train, y=y_train, steps=2000)
for var in nn.get_variable_names():
    print nn.get_variable_value(var)
    
#evaluate the model using validation set
results = nn.evaluate(x=X_train, y=y_train, steps=1)
type(results)
for key in sorted(results):
    print "%s:%s" % (key, results[key])
    
# Predict the outcome of test data using model
test = np.array([[100.4,21.5],[200.1,26.1]])
predictions = nn.predict(test)
predictions

