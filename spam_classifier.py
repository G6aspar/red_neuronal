import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Datos de entrenamiento (texto real simplificado)
textos = [
    "Ganaste un premio! Reclamalo ahora",
    "Trabajo desde casa con este truco secreto",
    "Actualización de tu cuenta bancaria",
    "Hola, ¿cómo estás?",
    "Reunión confirmada para mañana",
    "Recordatorio: turno médico hoy"
]

etiquetas = [1, 1, 1, 0, 0, 0]  # 1 = spam, 0 = no spam

# 2. Tokenización y padding
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(textos)

secuencias = tokenizer.texts_to_sequences(textos)
padded = pad_sequences(secuencias, padding='post')

# 3. Construcción del modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=100, output_dim=16, input_length=padded.shape[1]),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

modelo.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 4. Entrenamiento
modelo.fit(padded, np.array(etiquetas), epochs=30)

# 5. Prueba con nuevos textos
nuevos_textos = [
    "Reclamá tu dinero ahora",
    "Te espero en la oficina"
]

nuevas_secuencias = tokenizer.texts_to_sequences(nuevos_textos)
nuevas_padded = pad_sequences(nuevas_secuencias, padding='post', maxlen=padded.shape[1])

predicciones = modelo.predict(nuevas_padded)

for i, texto in enumerate(nuevos_textos):
    print(f"{texto} → Probabilidad de SPAM: {predicciones[i][0]:.2f}")
