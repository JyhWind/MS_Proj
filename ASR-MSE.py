import imp
import matplotlib.pyplot as plt;
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

def ASR_MSE(asr, mse):
    model = LinearRegression()
    model.fit(np.array(mse).reshape(-1,1), np.array(asr))
    x = range(6)
    y = model.predict(np.array(x).reshape(-1,1))

    feat = PolynomialFeatures(degree = 2)
    mse_2 = feat.fit_transform(np.array(mse).reshape(-1,1)) #将输入值进行降维
    model2 = LinearRegression()
    model2.fit(mse_2, np.array(asr))
    x_2 = range(6)
    y_2_input = feat.fit_transform(np.array(x).reshape(-1,1))
    y_2 = model2.predict(y_2_input)
    



    

    plt.figure()
    plt.plot(mse, asr, label = "Original data")
    plt.plot(x, y, 'red', label = "Linear Regression")
    plt.plot(x_2, y_2, 'green', label = "Quadratic Regression")
    plt.legend()

    plt.xlabel('MSE')
    plt.ylabel('ASR')
    plt.title('ASR-MSE Curve(FGSM)')
    
    plt.show()

mse = [1,2,3,4]
asr = [0.1, 0.25, 0.4, 0.8]
ASR_MSE(asr, mse)