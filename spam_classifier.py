import numpy as np
from keras.layers.core import Dense, Dropout
from keras.models import Sequential


def run():
    data = np.load('spam.npz')
    x_train, y_train = data['Xtrain'], data['ytrain']
    x_test, y_test = data['Xtest'], data['ytest']

    normalize(x_train)
    normalize(x_test)

    # After a few calibrations tests the ideal architecture was:
    # 3 hidden layers with 50 neurons each and ReLu activation with 0.3 dropout
    # Varying those params either does not make much difference or decrease accuracy
    model = Sequential()
    model.add(Dense(50, activation='relu', input_dim=57))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    # Using sigmoid in the final output layer since we are trying to approximate in between 0 and 1
    # Thus, cross-entropy is used for probabilistic loss calculation (in this case, of 2 classes)

    # Applying Nesterov momentum for the optimizer seems to perform relatively better than Adam or RMSProp
    # It also performs much better when compared to the traditional Stochastic Gradient Descent (SDG)
    model.compile(optimizer='nadam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # epochs close to 300 leads to better results with no significant changes after that
    # batch size around 100 seems ideal, otherwise it performs much worse
    print("\n === TRAINING === \n")
    model.fit(x_train, y_train, epochs=300, batch_size=100)

    print("\n === TEST RESULT === \n")
    score = model.evaluate(x_test, y_test, verbose=0, batch_size=100)
    print("loss: %s - acc: %s" % (score[0], score[1]))
    print("\n")


# Normalization method for training and testing data sets
# It ensures a global uniform distribution among all known feature values in between 0 and 1
# If not normalized, those input values will lead to a biased training and unpredictable patterns to be learned
#
# Note:
# Not necessarily all original and provided features are relevant. Adding too many irrelevant
# features will make learning harder and perform worse. Dropout helps mitigate this problem, but ideally
# feature selection techniques (or even manual selection by trail and error for small data sets) should be applied.
def normalize(data):
    max_word = np.amax(data[:, :48])
    max_char = np.amax(data[:, 48:54])
    max_avgl = np.amax(data[:, 54])
    max_longl = np.amax(data[:, 55])
    max_suml = np.amax(data[:, 56])

    for (i, j), v in np.ndenumerate(data):
        if j < 48:
            data[i, j] = v / max_word
        elif j < 54:
            data[i, j] = v / max_char
        elif j == 54:
            data[i, j] = v / max_avgl
        elif j == 55:
            data[i, j] = v / max_longl
        elif j == 56:
            data[i, j] = v / max_suml


if __name__ == '__main__':
    run()
