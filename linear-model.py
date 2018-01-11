import tensorflow as tf

# Model parameters
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([-1.0])

# Inputs and outputs
x = tf.placeholder(tf.float32)

y = tf.placeholder(tf.float32)

# Model
linear_model = W * x + b

# Squared error
squared_delta = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_delta)


# Optimization
gradient_optimizer = tf.train.GradientDescentOptimizer(0.01)
train = gradient_optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    for i in range (1000):
        session.run(train, {x: [1, 2, 3, 4], y: [2, 4, 6, 8]})

    print(session.run([W, b]))
