import keras

model = keras.models.load_model("NMA-Aniline_Step_3_model.h5")
model.summary()
model.layers.pop(0)

model.summary()

#newInput = Input(batch_shape = np.shape(sym_inp))
