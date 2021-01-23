from keras.losses import binary_crossentropy, mean_squared_error
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import Adam
from customLoss import MMD_Loss_func, adjust_binary_cross_entropy

class MMD_DRCN():
    def __init__(self, nSources, input_shape, nClass, ae=1, mmd=1, cls=1, sigmas=None, taskActivation='softmax', taskLoss='sparse_categorical_crossentropy'):
        self.input_shape = input_shape
        self.nClass = nClass
        self.hidden_layer = 1024
        self.taskActivation = taskActivation
        self.taskLoss = taskLoss
        self.mmd_loss = MMD_Loss_func(nSources, sigmas)

        self.weight_ae = ae
        self.weight_cls = cls
        self.weight_mmd = mmd

        self.E = self.convEncoder()
        self.D = self.convDecoder()(self.E.output)
        self.T = self.taskOut()(self.E.output)
    
    def convEncoder(self):
        model = Sequential(name='encoder')
        model.add(Conv2D(64, kernel_size = (3,3), padding='same', activation='relu', input_shape = self.input_shape))
        model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
        model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
        model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(self.hidden_layer, activation='relu'))
        model.add(Dropout(0.25, input_shape=[self.hidden_layer]))
        model.add(Dense(self.hidden_layer, activation='relu', name='encoder'))
        return model

    def convDecoder(self):
        model = Sequential(name='decoder')
        model.add(Dropout(0.25, input_shape=[self.hidden_layer, ]))
        model.add(Dense(self.hidden_layer, activation='relu', input_shape = [self.hidden_layer, ]))
        model.add(Dense(4096, activation='relu'))
        model.add(Reshape((4, 4, 256)))
        model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(UpSampling2D(size=(2,2)))
        model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(UpSampling2D(size=(2,2)))
        model.add(Conv2D(3, kernel_size=(3,3), padding='same', activation='relu'))
        return model

    def taskOut(self):
        model = Sequential(name='task')
        model.add(Dropout(0.25, input_shape=[self.hidden_layer, ]))
        model.add(Dense(self.hidden_layer, input_shape=[self.hidden_layer,], activation='relu'))
        model.add(Dense(self.nClass, activation=self.taskActivation))
        return model

    def makeDRCN(self):
        model = Model(self.E.input, [self.D, self.T])
        model.summary()
        model.compile(Adam(lr=10e-5), loss={'decoder': mean_squared_error, 'task': self.taskLoss}, loss_weights={'decoder': self.weight_ae, 'task': self.weight_cls})
        return model
    
    def makeMMD_DRCN(self):
        model = Model(self.E.input, [self.D, self.E.output, self.T])
        model.summary()
        model.compile(Adam(lr=10e-5), loss={'encoder': self.mmd_loss, 'decoder': mean_squared_error, 'task': self.taskLoss}, loss_weights={'encoder': self.weight_mmd, 'decoder': self.weight_ae, 'task': self.weight_cls})
        return model

    def makeBase(self):
        model = Model(self.E.input, self.T)
        model.summary()
        model.compile(Adam(lr=10e-5), loss={'task': self.taskLoss}, loss_weights={'task': self.weight_cls})
        return model


if __name__ == "__main__":
    model = MMD_DRCN(2, [17,17,3], 2)
    mmd_drcn = model.makeMMD_DRCN()
