import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from sklearn import model_selection
import numpy as np
import os
from tensorflow.python.ops import init_ops


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

#features = tf.random_uniform((5, 3 * 3 * 3), seed=1)
features = tf.constant([[1,2],[1,1]], dtype=tf.float32)
layer = layers.fully_connected(inputs=features, 
                               weights_initializer=tf.constant_initializer([1.0,1.0]), 
                               biases_initializer=tf.constant_initializer([1.0]),
                                                 num_outputs=2,
                                                 activation_fn=tf.nn.softmax)
targets = tf.constant([1,1,1,0], dtype=tf.float32)
outputs = tf.constant([0,0,0,1], dtype=tf.float32)
sq_loss1 = losses.mean_squared_error(outputs, targets)
log_loss1 = losses.log_loss(outputs, targets)

outputs = tf.constant([[100.0, -100.0, -100.0],
                      [-100.0, 100.0, -100.0],
                      [-100.0, -100.0, 100.0]])
targets = tf.constant([[0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0]])
sq_loss2 = losses.mean_squared_error(outputs, targets)

session = tf.Session()
session.run(tf.initialize_all_variables())
session.run(sq_loss2)
session.run(log_loss2)



# creating custom estimator
def model_function(features, targets, mode):
    
  #convert targets to one-hot vector representation   
  targets = tf.one_hot(targets, 2, 1, 0)

  # Configure the single layer perceptron model
  layer = layers.fully_connected(inputs=features,
                                                 num_outputs=2,
                                                 activation_fn=tf.sigmoid)
  outputs = learn.models.logistic_regression_zero_init(layer, targets)

  # Calculate loss using mean squared error
  loss = losses.mean_squared_error(outputs, targets)

  # Create an optimizer for minimizing the loss function
  optimizer = layers.optimize_loss(
      loss=loss,
      global_step=tf.contrib.framework.get_global_step(),
      learning_rate=0.001,
      optimizer="SGD")

  return {'labels':outputs}, loss, optimizer


#create custom estimator
#nn = learn.Estimator(model_fn=model_function, model_dir="/home/algo/m5")
nn = learn.Classifier(model_fn=model_function, n_classes=2, model_dir="/home/algo/m6"  )

#build the model
nn.fit(x=X_train, y=y_train, steps=2000)
for var in nn.get_variable_names():
    print nn.get_variable_value(var)
    
#evaluate the model using validation set
results = nn.evaluate(x=X_validation, y=y_validation, steps=1)
type(results)
for key in sorted(results):
    print "%s:%s" % (key, results[key])
    
# Predict the outcome of test data using model
X_test = np.array([[100.4,21.5],[200.1,26.1]], dtype=np.float32)
X_test.shape
predictions = nn.predict(X_test)
predictions

