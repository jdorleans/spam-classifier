# Installation

## Ubuntu
```
sudo apt install python3 python3-pip
pip3 install tensorflow keras
```

## Mac OS
```
brew install python3 pip3
pip3 install tensorflow keras
```

# Running

```
python3 spam_classifier.py
```
Output
```
Using TensorFlow backend.
 
 === TRAINING === 
 
Epoch 1/300
3065/3065 [==============================] - 0s - loss: 0.6432 - acc: 0.6036     
Epoch 2/300
3065/3065 [==============================] - 0s - loss: 0.4381 - acc: 0.8075     
Epoch 3/300
3065/3065 [==============================] - 0s - loss: 0.2890 - acc: 0.9008     
Epoch 4/300
3065/3065 [==============================] - 0s - loss: 0.2385 - acc: 0.9116     
...
Epoch 297/300
3065/3065 [==============================] - 0s - loss: 0.0419 - acc: 0.9834     
Epoch 298/300
3065/3065 [==============================] - 0s - loss: 0.0444 - acc: 0.9853     
Epoch 299/300
3065/3065 [==============================] - 0s - loss: 0.0424 - acc: 0.9883     
Epoch 300/300
3065/3065 [==============================] - 0s - loss: 0.0382 - acc: 0.9860     
 

 === TEST RESULT === 
 
loss: 0.323422917582 - acc: 0.94335937857
```
