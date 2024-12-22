import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

np.random.seed(42)

input_features = np.random.rand(1000, 1).astype(np.float32)
noise_component = np.random.normal(0, 2, size=(1000, 1)).astype(np.float32)
output_labels = 2 * input_features + 1 + noise_component

slope = tf.Variable(tf.random.normal([1]), name='slope')
intercept = tf.Variable(tf.zeros([1]), name='intercept')

learning_rate = 0.01
num_epochs = 20000
batch_size = 100

sgd_optimizer = tf.optimizers.SGD(learning_rate)

def calculate_loss(actual, predicted):
    return tf.reduce_mean(tf.square(actual - predicted))

loss_history = []

for epoch in range(num_epochs):
    random_indices = np.random.choice(len(input_features), batch_size)
    x_batch = input_features[random_indices]
    y_batch = output_labels[random_indices]

    with tf.GradientTape() as tape:
        predictions = slope * x_batch + intercept
        current_loss = calculate_loss(y_batch, predictions)
    gradients = tape.gradient(current_loss, [slope, intercept])
    sgd_optimizer.apply_gradients(zip(gradients, [slope, intercept]))

    loss_history.append(current_loss.numpy())

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch + 1}: Loss={current_loss.numpy():.4f}, "
              f"Slope={slope.numpy()[0]:.4f}, Intercept={intercept.numpy()[0]:.4f}")

print(f"Final model parameters: Slope={slope.numpy()[0]:.4f}, Intercept={intercept.numpy()[0]:.4f}")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(input_features, output_labels, label='Training Data', alpha=0.5)
plt.plot(input_features, slope.numpy() * input_features + intercept.numpy(),
         color='red', label='Regression Line')
plt.title('Linear Regression Fit')
plt.xlabel('Input Feature')
plt.ylabel('Output Label')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(num_epochs), loss_history, color='blue', label='Loss Curve')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()