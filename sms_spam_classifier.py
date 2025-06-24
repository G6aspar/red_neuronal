import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Cargar dataset desde archivo TSV (sin encabezado)
df = pd.read_csv('spam.csv', sep='\t', header=None, names=['label', 'text'])
df['label_enc'] = df['label'].map({'ham': 0, 'spam': 1})

# División entrenamiento/test
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label_enc'], test_size=0.2, random_state=42
)

# 2. Tokenizador
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Secuencias y padding
seq_train = tokenizer.texts_to_sequences(X_train)
seq_test = tokenizer.texts_to_sequences(X_test)

max_len = 100
X_tr = pad_sequences(seq_train, maxlen=max_len, padding='post')
X_te = pad_sequences(seq_test, maxlen=max_len, padding='post')

# 3. Modelo Embedding + LSTM
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=64, input_length=max_len),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. Entrenamiento
history = model.fit(
    X_tr, y_train, epochs=5, batch_size=64,
    validation_data=(X_te, y_test)
)

# 5. Evaluación
loss, acc = model.evaluate(X_te, y_test)
print(f"\nTest Accuracy: {acc:.3f}, Loss: {loss:.3f}")

# 6. Prueba con ejemplos nuevos
samples = ["Reclamá tu premio ahora!",
           "¿Cuándo es la reunión de mañana?"]
seq = tokenizer.texts_to_sequences(samples)
padded = pad_sequences(seq, maxlen=max_len, padding='post')
pred = model.predict(padded)

for txt, p in zip(samples, pred):
    etiqueta = "SPAM" if p>0.5 else "HAM"
    print(f"{txt} → {etiqueta} ({p[0]:.2f})")
