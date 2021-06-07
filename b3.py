import pandas as pd 
import numpy as np
import math
import matplotlib.pyplot as plt

# h(Xi) = b[0] + b[1]*ageArray 
# e[i] = bmiArray[i] - h[i]

def coef(x, y):
    N = np.size(x)
 
    mean_of_x = np.mean(x)
    mean_of_y = np.mean(y)
 
    SS_xy = np.sum(y*x) - N*mean_of_y*mean_of_x
    SS_xx = np.sum(x*x) - N*mean_of_x*mean_of_x
 
    b_1 = SS_xy / SS_xx
    b_0 = mean_of_y - b_1*mean_of_x
 
    return (b_0,b_1)

def plot_regression_line(x, y ,b):

    plt.scatter(x, y, color = "m",
                marker = "o", s = 30)

    # predicted response vector
    y_pred = b[0] + b[1]*x

    # plotting the regression line
    plt.plot(x, y_pred, color = "g")

    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')

    # function to show plot
    plt.show()
    return y_pred



def main():
    data = pd.read_csv("healthcare-dataset-stroke-data.csv")
    ageArray = data["age"]
    bmiArray = data["bmi"]
    print("size of arrays: ", len(ageArray))
    missing_value = float('nan')
    
    #h = [len(ageArray)]
    #e = [len(ageArray)]

    b = coef(ageArray,bmiArray)
    bmi = plot_regression_line(ageArray, bmiArray, b)
    print(bmi)
    print(bmiArray)
    for i in range(0,len(ageArray)):
        if(math.isnan(bmiArray[i])):
            bmiArray[i]=round(bmi[i],1)
    print(bmiArray)
    bmiArray.to_csv(r'./B3.csv', index = False)
    
    
if __name__ == "__main__":
    main()