import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import config

# Load the dataset
train_ds, test_ds, _ = config.get_datasets()

# change the format of the class weights to a dictionary
class_weights = {0: 1.3772893772893773, 1: 1.0662885501595178, 2: 1.1489686783804431, 3: 0.840928152082751, 4: 0.8769679300291545, 5: 0.9591836734693877, 6: 0.9142857142857143}

print(f"Generated class weights for an imbalanced dataset: {class_weights}")

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = keras.Sequential()
model.add(keras.Input(shape=(config.image_size[0], config.image_size[1], 1)))

# Preprocessing
model.add(keras.layers.RandomTranslation(0.1, 0.1))
model.add(keras.layers.RandomBrightness(0.1))
model.add(keras.layers.RandomContrast(0.1))

# Rescale the pixel values
model.add(keras.layers.Rescaling(1./255))

# Convolutional layers
model.add(keras.layers.Conv2D(16,3, padding='same', activation='relu'))
model.add(keras.layers.Conv2D(16,3, padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Conv2D(32,3, padding='same', activation='relu'))
model.add(keras.layers.Conv2D(32,3, padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Conv2D(64,3, padding='same', activation='relu'))
model.add(keras.layers.Conv2D(64,3, padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Flatten())

# Dense Layers
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(config.n_classes, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), 
                     loss='categorical_crossentropy', 
                     metrics=[keras.metrics.CategoricalAccuracy(),
                              keras.metrics.AUC(),
                              keras.metrics.F1Score(), 
                              keras.metrics.Precision(),
                              keras.metrics.Recall()])


model.summary()

checkpoint_filepath = 'models/instrument-recognition.checkpoint.best.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_categorical_accuracy',
    mode='max',
    save_best_only=True)

# Model is saved at the end of every epoch, if it's the best seen so far.
history = model.fit(train_ds, validation_data=test_ds, epochs=200, class_weight=class_weights, callbacks=[model_checkpoint_callback])

# Save the model
model.save_weights("models/instrument-recognition.weights.h5")

np.save('models/history.npy', history.history)
