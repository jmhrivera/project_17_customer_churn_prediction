# import torch

# # Verificar si CUDA está disponible
# if torch.cuda.is_available():
#     device = torch.device("cuda")          # Seleccionar GPU
#     print(f"CUDA está disponible! Usando la GPU: {torch.cuda.get_device_name(0)}")
# else:
#     device = torch.device("cpu")           # Si no hay GPU, usar CPU
#     print("CUDA no está disponible. Usando CPU.")

# # Ejemplo de tensor en el dispositivo seleccionado
# x = torch.randn(3, 3).to(device)

# # Mostrar información del tensor y dispositivo
# print(f"Tensor x:\n{x}")
# print(f"Dispositivo de x:\n{x.device}")


##########################
#  import tensorflow as tf

# # Verificar la disponibilidad de GPUs
# gpu_name = tf.test.gpu_device_name()
# gpu_available = tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

# print(f"Nombre del GPU: {gpu_name}")
# print(f"GPU disponible: {gpu_available}")

#############################
import tensorflow as tf

# Verificar la disponibilidad de GPUs
print("GPUs disponibles:")
print(tf.config.list_physical_devices('GPU'))

# Crear un modelo simple
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Cargar datos de ejemplo (MNIST)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocesar datos
x_train, x_test = x_train / 255.0, x_test / 255.0

# Entrenar el modelo usando GPU
with tf.device('/GPU:0'):
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Evaluar el modelo
model.evaluate(x_test, y_test)
