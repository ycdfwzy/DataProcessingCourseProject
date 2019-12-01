import time
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from sklearn.utils import shuffle
import pandas as pd

dataPath = '../data/classification_data.csv'


BS = 12
NN = 11500
n_split = 5

def classification(X, y, hls):
    # clf = KNeighborsClassifier(2)
    # clf = tree.DecisionTreeClassifier()
    # clf = svm.LinearSVC()
    # clf = MultinomialNB(alpha=0.5)
    clf = MLPClassifier(hidden_layer_sizes=hls, learning_rate_init=0.001, activation='relu', \
                        solver='adam', alpha=0.0001, max_iter=30000)
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    return scores.mean()


def get_dataset(file_path, columns_to_use, K_classiication):
    df = pd.read_csv(file_path, usecols=columns_to_use)
    df = shuffle(df)
    data_x = df.iloc[:,0:178].values
    data_y = df.iloc[:,178:179].values
    data_y = data_y - 1
    if K_classiication == 2:
        for i in range(np.size(data_y, 0)):
            if data_y[i][0] != 0:
                data_y[i] = [1]
    return data_x, data_y


# def get_dataset(file_path, columns_to_use, label_colunm):
#     dataset = tf.data.experimental.make_csv_dataset(
#         file_path,
#         batch_size=BS,
#         select_columns=columns_to_use,
#         label_name=label_colunm,
#         shuffle=False
#     )
#     return dataset

from tensorflow.keras.layers import Conv1D, AveragePooling1D, BatchNormalization, Activation, Input, Dense, MaxPooling1D, GlobalAveragePooling1D, Concatenate
from tensorflow.keras import Model

def get_model(K_classiication):

    def denseblock(tensor, filters, layer_num, name):
        rtn = tensor
        l = [tensor]
        for i in range(layer_num):
            if(len(l)>=2):
                print(l[0:2])
                inp=Concatenate(name = '''si{}_concat{}'''.format(name,i))(l[0:2])
                # inp = l[-1]
            else:
                inp=tensor
            conv = Conv1D(filters=filters, kernel_size=3, strides=1, padding= 'same', name='''si{}_conv{}'''.format(name, i))(inp)
            bn = BatchNormalization(name='''si{}_bn{}'''.format(name,i))(conv)
            ac = Activation('relu', name = '''si{}_relu{}'''.format(name, i))(bn)
            # l.append(ac)
            # l.append(ac)

            # print(l)
            # rtn = Concatenate(name = '''si{}_concat{}'''.format(name, i))(l)
            # return rtn
            if i != layer_num-1:
                bottleneck = Conv1D(filters=filters//4, kernel_size=1, padding = 'same', name = '''si{}_bt{}'''.format(name,i))(ac)
                l.append(bottleneck)
            # l.append(ac)
            rtn = ac
        return rtn

    if K_classiication == 5:
        input = Input((178,))
        # block
        x = tf.expand_dims(input, axis=-1 )
        # x1 = Conv1D(kernel_size=1, filters=4, padding = 'same')(x)
        # x2 = Conv1D(kernel_size=3, filters=4, padding = 'same')(x)
        # x = Concatenate()([x1,x2])
        # x = denseblock(x, 8, 4, 'db1')
        # x = denseblock(x, 16, 7, 'db2')
        # x = denseblock(x, 24, 11, 'db3')
        # x = denseblock(x, 32, 13, 'db4')

        x = Conv1D(kernel_size=3, filters=4, padding = 'same')(x)
        x = Activation('relu')(x)
        # x = AveragePooling1D(2)(x)
        x = BatchNormalization()(x)

        x = Conv1D(kernel_size=3, filters=6, padding = 'same')(x)
        x = Activation('relu')(x)
        x = AveragePooling1D(2)(x)
        x = BatchNormalization()(x)

        x = Conv1D(kernel_size=3, filters=8, padding = 'same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)


        x = Conv1D(kernel_size=3, filters=12, padding = 'same')(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(2)(x)
        x = BatchNormalization()(x)

        x = Conv1D(kernel_size=3, filters=14, padding = 'same')(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(2)(x)
        x = BatchNormalization()(x)

        x = Conv1D(kernel_size=3, filters=22, padding = 'same')(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(2, padding = 'same')(x)
        x = BatchNormalization()(x)

        x = Conv1D(kernel_size=3, filters=26, padding = 'same')(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(2)(x)
        x = BatchNormalization()(x)

        x = Conv1D(kernel_size=2, strides = 2, filters = 28)(x)
        x = Activation('relu')(x)


        x = GlobalAveragePooling1D()(x)
        # x = tf.expand_dims(x, axis = 0)
        x = Dense(K_classiication, activation='softmax')(x)
        # model = Model(input, x)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(178,)),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(K_classiication, activation='softmax')
        ])
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(178,)),
            # tf.keras.layers.Dense(170, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(K_classiication, activation='softmax')
        ])
        # model = tf.keras.Sequential([
        #     tf.keras.layers.Input(shape=(178,)),
        #     tf.keras.layers.Dense(160, activation='relu'),
        #     tf.keras.layers.Dropout(0.2),
        #     tf.keras.layers.Dense(60, activation='relu'),
        #     tf.keras.layers.Dropout(0.2),
        #     tf.keras.layers.Dense(6, activation='relu'),
        #     tf.keras.layers.Dense(K_classiication, activation='softmax')
        # ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main(K_classiication):
    columns_to_use = []
    for i in range(1, 179):
        columns_to_use.append("X" + str(i))
    columns_to_use.append("y")
    X, Y = get_dataset(dataPath, columns_to_use, K_classiication)

    scores = []
    for train_index, test_index in KFold(n_split).split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        model = get_model(K_classiication)
        model.fit(x_train, y_train, batch_size=32, epochs=40, verbose=0)

        scores.append(model.evaluate(x_test, y_test, verbose=0)[-1])
    print(str(K_classiication)+'-classification accuracy: ', np.array(scores).mean())

if __name__ == '__main__':
    start = time.time()
    main(5)
    end = time.time()
    print('totally cost for 5-classification: ', end - start)

    start = time.time()
    main(2)
    end = time.time()
    print('totally cost for 2-classification: ', end - start)

