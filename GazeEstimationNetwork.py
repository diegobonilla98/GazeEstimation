# from HourglassNet import create_hourglass_network, euclidean_loss
from BoniDL import utils, losses
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from DataLoader import DataLoader
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import CoolUnet

utils.allow_gpu_growth()

# model = create_hourglass_network(num_classes=1, num_stacks=2, num_channels=64, inres=(64, 64))
model = CoolUnet.custom_unet(input_shape=(64, 64, 1), use_attention=True, filters=32)
opt = Adam(lr=5e-4)
model.compile(optimizer=opt, loss=losses.bce_dice_loss, metrics=["accuracy"])
model.summary()
plot_model(model, show_shapes=True)

train_generator = DataLoader((64, 64), (64, 64), 16)
callbacks = [EarlyStopping(monitor='loss', min_delta=0.0001, patience=7, verbose=1, restore_best_weights=True),
             ReduceLROnPlateau(monitor='loss', min_delta=0.0002, patience=2, verbose=1)]
model.fit_generator(generator=train_generator, epochs=1000, callbacks=callbacks)
model.save('model_att.h5', include_optimizer=False)
