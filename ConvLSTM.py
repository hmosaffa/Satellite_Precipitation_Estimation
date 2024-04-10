import tensorflow as tf
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv2D, Dropout

def build_convlstm_model(sequence_length, input_shape=(sequence_length, 128, 128, 6)):
    inp = Input(shape=input_shape)  

    convlstm1 = ConvLSTM2D(32, kernel_size=(5, 5), padding="same", activation="relu", return_sequences=True)(inp)
    bn1 = BatchNormalization()(convlstm1)

    convlstm2 = ConvLSTM2D(64, kernel_size=(5, 5), padding="same", activation="relu", return_sequences=True)(bn1)
    bn2 = BatchNormalization()(convlstm2)

    convlstm3 = ConvLSTM2D(128, kernel_size=(5, 5), padding="same", activation="relu", return_sequences=True)(bn2)
    bn3 = BatchNormalization()(convlstm3)

    conv6 = Conv2D(32, kernel_size=(5, 5), padding="same", activation="relu")(bn3)
    bn6 = BatchNormalization()(conv6)

    dropout1 = Dropout(0.4)(bn6)
    conv7 = Conv2D(16, kernel_size=(5, 5), padding="same", activation="relu")(dropout1)
    bn7 = BatchNormalization()(conv7)
    conv8 = Conv2D(8, kernel_size=(5, 5), padding="same", activation="relu")(bn7)
    bn8 = BatchNormalization()(conv8)

    out = Conv2D(1, kernel_size=(5, 5), padding="same", activation="relu")(bn8)

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(loss="mse", optimizer="sgd")
    return model

sequence_length = 3
inp_data = np.reshape(inp_data, (-1, sequence_length, 128, 128, 6))
out_data = np.reshape(out_data, (-1, sequence_length, 128, 128, 1))  

model = build_convlstm_model(sequence_length)
model.summary()  # Optional: Print model summary to verify the architecture

model.fit(inp_data, out_data, epochs=10, batch_size=64, verbose=True)
