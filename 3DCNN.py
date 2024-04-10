import tensorflow as tf

def build_3dcnn_model(sequence_length, input_shape=(sequence_length, 128, 128, 6)):
    # Input layer
    inp = tf.keras.layers.Input(shape=input_shape)
    
    # Convolutional layers
    conv1 = tf.keras.layers.Conv3D(32, kernel_size=(2, 5, 5), padding="same", activation="relu")(inp)
    bn1 = tf.keras.layers.BatchNormalization()(conv1)
    pool1 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(bn1)

    conv2 = tf.keras.layers.Conv3D(64, kernel_size=(2, 5, 5), padding="same", activation="relu")(pool1)
    bn2 = tf.keras.layers.BatchNormalization()(conv2)

    conv3 = tf.keras.layers.Conv3D(128, kernel_size=(2, 5, 5), padding="same", activation="relu")(bn2)
    bn3 = tf.keras.layers.BatchNormalization()(conv3)

    conv6 = tf.keras.layers.Conv3D(256, kernel_size=(2, 5, 5), padding="same", activation="relu")(bn3)
    bn6 = tf.keras.layers.BatchNormalization()(conv6)

    # Upsampling and concatenation layers
    upsample2 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(bn6)
    conv7 = tf.keras.layers.Conv3D(32, kernel_size=(2, 5, 5), padding="same", activation="relu")(upsample2)
    bn7 = tf.keras.layers.BatchNormalization()(conv7)

    combine2 = tf.keras.layers.Concatenate()([bn1, bn7])

    conv8 = tf.keras.layers.Conv3D(32, kernel_size=(2, 5, 5), padding="same", activation="relu")(combine2)
    bn8 = tf.keras.layers.BatchNormalization()(conv8)

    # Dropout layer
    dropout1 = tf.keras.layers.Dropout(0.4)(bn8)

    conv9 = tf.keras.layers.Conv3D(16, kernel_size=(2, 5, 5), padding="same", activation="relu")(dropout1)
    bn9 = tf.keras.layers.BatchNormalization()(conv9)

    conv10 = tf.keras.layers.Conv3D(8, kernel_size=(2, 5, 5), padding="same", activation="relu")(bn9)
    bn10 = tf.keras.layers.BatchNormalization()(conv10)

    # Output layer
    out = tf.keras.layers.Conv3D(1, kernel_size=(2, 5, 5), padding="same", activation="relu")(bn10)

    # Model compilation
    model = tf.keras.models.Model(inputs=inp, outputs=out) 
    model.compile(loss="mse", optimizer="sgd")
    
    return model

model = build_3dcnn_model(sequence_length=sequence_length)
model.fit(inp_data, out_data, epochs=50, batch_size=64, verbose=True)
