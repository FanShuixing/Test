import keras
import numpy as np 

model = keras.models.load_model('./model/vgg_3_200_mse.model')
x_predict = np.load('./train_512_fenlei123_vaild.npy')
x_predict = x_predict.reshape((76, 512, 512, 23))
y_predict = model.predict(x_predict)
print(y_predict.shape)
y_true = np.load('./label_512_fenlei123_vaild.npy')
m = 0

for i in range(y_predict.shape[0]):
    #print(y_predict[i],y_true[i])
    y_p = np.zeros(3)
    postition_predict = np.argwhere(y_predict[i] == np.max(y_predict[i]))
    postition_true = np.argwhere(y_true[i] == np.max(y_true[i]))
    print(postition_predict, postition_true)

    if postition_predict[0][0] ==postition_true[0][0]:
        m += 1
        print(y_predict[i],y_true[i])

acc = m/(y_predict.shape[0])
print('acc:',acc)

