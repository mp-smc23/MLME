import tensorflow as tf

# Define the labels 
# Example encoding of a label [1, 0, 0, 0, 0, 0]
irmas_labels = ["cel", "cla", "flu", "gac", "sax", "tru", "vio"]
label_names = ["cello", "clarinet", "flute", "guitar", "saxophone", "trumpet", "violin"]
n_classes = len(label_names)
data_dir = "datasets/IRMAS-TrainingData-images"

image_size = (128, 128)
batch_size = 32

def get_datasets():
    # Load the data
    full_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed = 123,
        class_names=irmas_labels
    )
    
    DATASET_SIZE = full_ds.cardinality().numpy()
    
    train_size = int(0.8 * DATASET_SIZE)
    test_size = int(0.1 * DATASET_SIZE)

    train_ds = full_ds.take(train_size)
    test_ds = full_ds.skip(train_size)
    val_ds = test_ds.skip(test_size)
    test_ds = test_ds.take(test_size)
    
    return train_ds, test_ds, val_ds