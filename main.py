import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
import sys
import random
import io

path = keras.utils.get_file(
    fname="don_quijote.txt",
    origin="https://onedrive.live.com/download?cid=C506CF0A4F373B0F&resid=C506CF0A4F373B0F%219424&authkey=AH0gb-qSo5Xd7Io"
)

#Read document
with open(path, encoding="utf8") as f:
  text = f.read().lower()

#map char
chars=sorted(list(set(text)))
char_indices = dict((c,i) for i, c in enumerate(chars))
indice_char = dict((i, c) for i, c in enumerate(chars))


#get sequense input and char predict
'''
Example sequence
sequences = ["Don Q", "on Qu", "n Qui", " Quij", "Quijo", "uijot"]
next_chars = ['u', 'i', 'j', 'o', 't', 'e']
'''
SEQ_LENGTH = 35
step=3
rawX = []
rawy = []

for i in range(0, len(text) - SEQ_LENGTH, step):
    rawX.append(text[i: i+SEQ_LENGTH])
    rawy.append(text[i+SEQ_LENGTH])


#Fix proble, memory. select num Max
MAX_SEQUENCES = 200000

perm = np.random.permutation(len(rawX)) #Permutar aleatoriamente una secuencia, o devolver un rango permutado.
rawX, rawy = np.array(rawX), np.array(rawy)
rawX, rawy = rawX[perm], rawy[perm]
rawX, rawy = list(rawX[:MAX_SEQUENCES]), list(rawy[:MAX_SEQUENCES])

print(len(rawX))

#Create array for model
X = np.zeros((len(rawX), SEQ_LENGTH , len(chars)))
y = np.zeros((len(rawX), len(chars)))

for i, sentence in enumerate(rawX):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[rawy[i]]] = 1

#Define model
model= Sequential()
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(chars))))
model.add(Dropout(0.2))
model.add(Dense(len(chars), activation= "softmax"))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def sample(probs, temperature=1.0):
    """Nos da el índice del elemento a elegir según la distribución
    de probabilidad dada por probs.

    Args:
      probs es la salida dada por una capa softmax:
        probs = model.predict(x_to_predict)[0]

      temperature es un parámetro que nos permite obtener mayor
        "diversidad" a la hora de obtener resultados.

        temperature = 1 nos da la distribución normal de softmax
        0 < temperature < 1 hace que el sampling sea más conservador,
          de modo que sampleamos cosas de las que estamos más seguros
        temperature > 1 hace que los samplings sean más atrevidos,
          eligiendo en más ocasiones clases con baja probabilidad.
          Con esto, tenemos mayor diversidad pero se cometen más
          errores.
    """
    # Cast a float64 por motivos numéricos
    probs = np.asarray(probs).astype('float64')

    # logaritmo de probabilidades y aplicamos reducción
    # por temperatura.
    probs = np.log(probs) / temperature

    # Volvemos a aplicar exponencial y normalizamos de nuevo
    exp_probs = np.exp(probs)
    probs = exp_probs / np.sum(exp_probs)

    # Hacemos el sampling dadas las nuevas probabilidades
    # de salida (ver doc. de np.random.multinomial)
    samples = np.random.multinomial(1, probs, 1)
    return np.argmax(samples)

TEMPERATURES_TO_TRY = [0.2] #, 0.5, 1.0, 1.2]
GENERATED_TEXT_LENGTH = 300

def generate_text(seed_text, model, length=300, temperature=1, max_length=30):
    """Genera una secuencia de texto a partir de seed_text utilizando model.

    La secuencia tiene longitud length y el sampling se hace con la temperature
    definida.
    """

    # Aquí guardaremos nuestro texto generado, que incluirá el
    # texto origen
    generated = seed_text

    # Utilizar el modelo en un bucle de manera que generemos
    # carácter a carácter. Habrá que construir los valores de
    # X_pred de manera similar a como hemos hecho arriba, salvo que
    # aquí sólo se necesita una oración
    # Nótese que el x que utilicemos tiene que irse actualizando con
    # los caracteres que se van generando. La secuencia de entrada al
    # modelo tiene que ser una secuencia de tamaño SEQ_LENGTH que
    # incluya el último caracter predicho.

    ### TU CÓDIGO AQUÍ
    prediction = []

    #textReturn
    textReturn = ""

    for i in range(length):
        # Make numpy array to hold seed
        X = np.zeros((1, len(generated), len(chars) ))

        # Set one-hot vectors for seed sequence
        for t, char in enumerate(seed_text):
            X[0, t, char_indices[char]] = 1

        # Generate prediction for next character
        preds = model.predict(X, verbose=0)[0]
        # Choose a character from the prediction probabilities
        next_index = sample(preds,0.2)
        next_char = indice_char[next_index]

        prediction.append(next_char)
        # Add the predicted character to the seed sequence so the next prediction
        # includes this character in it's seed.
        #generated += next_char
        seed_text = seed_text[1:] + next_char
        #add to textReturn predict
        textReturn = textReturn+next_char
        print(next_char, end= " ");
        # Flush so we can see the prediction as it's generated
        sys.stdout.flush()

    prediction = ''.join(prediction)
    sys.stdout.flush()

    ### FIN DE TU CÓDIGO
    return textReturn


def on_epoch_end(epoch, logs):
  print("\n\n\n")

  # Primero, seleccionamos una secuencia al azar para empezar a predecir
  # a partir de ella
  start_pos = random.randint(0, len(text) - SEQ_LENGTH - 1)
  seed_text = text[start_pos:start_pos + SEQ_LENGTH]
  for temperature in TEMPERATURES_TO_TRY:
    print("\nEpoch: {}, Loss: {:.4f}, Accuracy: {:.2f}%".format(
          epoch + 1, logs['loss'], logs['accuracy'] * 100))

    generated_text = generate_text(seed_text, model,
                                   GENERATED_TEXT_LENGTH, temperature)
    print("Seed: {}".format(seed_text))
    print("Texto generado: {}".format(generated_text))


generation_callback = LambdaCallback(on_epoch_end=on_epoch_end)

#Training model
history = model.fit(X, y, batch_size=128, epochs=50, verbose=0, callbacks=[generation_callback],validation_split=0.2)
history.history['accuracy']
#model save
model.save("model.h5")