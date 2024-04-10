import tensorflow as tf

def build_2dcnn_model(input_shape=(128, 128, 6)):
    # Input layer
    inp = tf.keras.layers.Input(shape=input_shape)
    
    # Convolutional layers
    conv1 = tf.keras.layers.Conv2D(32, kernel_size=5, padding="same", activation="relu")(inp)
    bn1 = tf.keras.layers.BatchNormalization()(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(bn1)

    conv2 = tf.keras.layers.Conv2D(64, kernel_size=5, padding="same", activation="relu")(pool1)
    bn2 = tf.keras.layers.BatchNormalization()(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = tf.keras.layers.Conv2D(128, kernel_size=5, padding="same", activation="relu")(pool2)
    bn3 = tf.keras.layers.BatchNormalization()(conv3)

    conv5 = tf.keras.layers.Conv2D(64, kernel_size=5, padding="same", activation="relu")(bn3)
    bn5 = tf.keras.layers.BatchNormalization()(conv5)

    # Upsampling and concatenation layers
    upsam1 = tf.keras.layers.UpSampling2D(size=(2, 2))(bn5)
    combine1 = tf.keras.layers.Concatenate()([bn2, upsam1])

    conv6 = tf.keras.layers.Conv2D(256, kernel_size=5, padding="same", activation="relu")(combine1)
    bn6 = tf.keras.layers.BatchNormalization()(conv6)

    upsam2 = tf.keras.layers.UpSampling2D(size=(2, 2))(bn6)
    conv7 = tf.keras.layers.Conv2D(32, kernel_size=5, padding="same", activation="relu")(upsam2)
    bn7 = tf.keras.layers.BatchNormalization()(conv7)

    combine2 = tf.keras.layers.Concatenate()([bn1, bn7])

    conv8 = tf.keras.layers.Conv2D(32, kernel_size=5, padding="same", activation="relu")(combine2)
    bn8 = tf.keras.layers.BatchNormalization()(conv8)

    # Dropout layer
    dropout1 = tf.keras.layers.Dropout(0.4)(bn8)

    conv9 = tf.keras.layers.Conv2D(16, kernel_size=5, padding="same", activation="relu")(dropout1)
    bn9 = tf.keras.layers.BatchNormalization()(conv9)

    conv10 = tf.keras.layers.Conv2D(8, kernel_size=5, padding="same", activation="relu")(bn9)
    bn10 = tf.keras.layers.BatchNormalization()(conv10)

    # Output layer
    out = tf.keras.layers.Conv2D(1, kernel_size=5, padding="same", activation="relu")(bn10)

    # Model compilation
    model = tf.keras.models.Model(inputs=inp, outputs=out) 
    model.compile(loss="mse", optimizer="sgd")
    
    return model

model = build_2dcnn_model()
model.fit(inp_data, out_data, epochs=100, batch_size=64, verbose=True)
