import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, Subtract, Concatenate, Dropout

def build_siamese_diff_model(input_shape1=(128, 128, 5), input_shape2=(128, 128, 1), input_shape3=(128, 128, 1)):
    # Define input layers
    inp1 = Input(shape=input_shape1)
    inp2 = Input(shape=input_shape2)
    inp3 = Input(shape=input_shape3)

    # Model for inp2
    conv1_2 = Conv2D(32, kernel_size=5, padding="same", activation="relu")(inp2)
    bn1_2 = BatchNormalization()(conv1_2)
    pool1_2 = MaxPooling2D(pool_size=(2, 2))(bn1_2)
    conv2_2 = Conv2D(64, kernel_size=5, padding="same", activation="relu")(pool1_2)
    bn2_2 = BatchNormalization()(conv2_2)

    # Model for inp3
    conv1_3 = Conv2D(32, kernel_size=5, padding="same", activation="relu")(inp3)
    bn1_3 = BatchNormalization()(conv1_3)
    pool1_3 = MaxPooling2D(pool_size=(2, 2))(bn1_3)
    conv2_3 = Conv2D(64, kernel_size=5, padding="same", activation="relu")(pool1_3)
    bn2_3 = BatchNormalization()(conv2_3)

    abs_diff_2 = Subtract()([bn2_2, bn2_3])
    abs_diff_1 = Subtract()([bn1_2, bn1_3])

    # Model for inp1
    conv1 = Conv2D(32, kernel_size=5, padding="same", activation="relu")(inp1)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)
    conv2 = Conv2D(64, kernel_size=5, padding="same", activation="relu")(pool1)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)
    conv3 = Conv2D(128, kernel_size=5, padding="same", activation="relu")(pool2)
    bn3 = BatchNormalization()(conv3)
    conv5 = Conv2D(64, kernel_size=5, padding="same", activation="relu")(bn3)
    bn5 = BatchNormalization()(conv5)
    upsam1 = UpSampling2D(size=(2, 2))(bn5)

    bn2_con = Concatenate()([abs_diff_2, bn2])

    combine1 = Concatenate()([bn2_con, upsam1])

    conv6 = Conv2D(256, kernel_size=5, padding="same", activation="relu")(combine1)
    bn6 = BatchNormalization()(conv6)
    upsam2 = UpSampling2D(size=(2, 2))(bn6)
    conv7 = Conv2D(32, kernel_size=5, padding="same", activation="relu")(upsam2)
    bn7 = BatchNormalization()(conv7)

    bn7_con = Concatenate()([abs_diff_1, bn1])

    combine2 = Concatenate()([bn1, bn7_con])
    conv8 = Conv2D(32, kernel_size=5, padding="same", activation="relu")(combine2)
    bn8 = BatchNormalization()(conv8)
    dropout1 = Dropout(0.4)(bn8)
    conv9 = Conv2D(16, kernel_size=5, padding="same", activation="relu")(dropout1)
    bn9 = BatchNormalization()(conv9)
    conv10 = Conv2D(8, kernel_size=5, padding="same", activation="relu")(bn9)
    bn10 = BatchNormalization()(conv10)
    out = Conv2D(1, kernel_size=5, padding="same", activation="relu")(bn10)

    model = tf.keras.models.Model(inputs=[inp1, inp2, inp3], outputs=out)
    model.compile(loss="mse", optimizer="sgd")
    return model

model = build_siamese_diff_model()
model.summary()  
model.fit([inp_data1, inp_data2, inp_data3], out_data, epochs=50, batch_size=64, verbose=True)
