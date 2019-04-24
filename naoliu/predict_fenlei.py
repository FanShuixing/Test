import keras
import numpy as np 

model = keras.models.load_model('./model/resnet_300ep_256_cc_cv_new.model')
x_predict = np.load('./x_vaild_new.npy')
print(x_predict.shape)

#filepath = './meta_train.txt'
#x_predict = x_predict.reshape((1885, 256, 256, 23))
y_predict = model.predict(x_predict)
print('y_predict',y_predict.shape)
y_true = np.load('./y_vaild_new.npy')

m = 0
n = 0
a = 0
b = 0
c = 0
d = 0

for i in range(y_predict.shape[0]):
    if int(y_true[i])==0:
        c += 1
        print(y_predict[i], np.argmax(y_predict[i]), y_true[i])
        if np.argmax(y_predict[i])==int(y_true[i]):
            d += 1    
            
    if int(y_true[i])==1:
        n += 1
        print(y_predict[i], np.argmax(y_predict[i]), y_true[i])
        if np.argmax(y_predict[i])==int(y_true[i]):
            m += 1
            
    elif int(y_true[i])==2:
        a += 1
        print(y_predict[i], np.argmax(y_predict[i]), y_true[i])
        if np.argmax(y_predict[i])==int(y_true[i]):
            b += 1   


'''
for i in range(y_predict.shape[0]):
    #print(y_predict[i],y_true[i])
    y_p = np.zeros(3)
    postition_predict = np.argwhere(y_predict[i] == np.max(y_predict[i]))
    postition_true = np.argwhere(y_true[i] == np.max(y_true[i]))
    print(postition_predict, postition_true)

    if postition_predict[0][0] ==postition_true[0][0]:
        m += 1
        print(y_predict[i],y_true[i])
    if np.argmax(y_predict[i])==int(y_true[i]):
        print(y_predict[i], np.argmax(y_predict[i]), y_true[i])
        m += 1
    if int(y_true[i])!=0:
        n += 1
        print(y_predict[i], np.argmax(y_predict[i]), y_true[i])
        if np.argmax(y_predict[i])==int(y_true[i]):
            m += 1

    if np.argmax(y_predict[i])==int(y_true[i]):
        print(y_predict[i], np.argmax(y_predict[i]), y_true[i])
        m += 1  
        
'''
            
            
#acc = m/(y_predict.shape[0])
#print(acc)
print(c, d, n, m, a, b)
acc_0 = d/c
acc_1 = m/n
acc_2 = b/a
total_right = d+m+b
total = c+n+a
total_acc = total_right/total
print('total_acc', total_acc, 'acc_0', acc_0, 'acc_1:',acc_1, 'acc_2:',acc_2)

