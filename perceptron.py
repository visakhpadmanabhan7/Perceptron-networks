import sys

import numpy as np
import csv
import pandas as pd
def create_nueral_network(x_train,y_train,learning_rate,iterations):
    w_a_h1 = -0.3
    w_b_h1 = 0.4
    w_a_h2 = -0.1
    w_b_h2 = -0.4
    w_a_h3 = 0.2
    w_b_h3 = 0.1
    w_h1_o = 0.1
    w_h2_o = 0.3
    w_h3_o = -0.4
    w_bias_h1 = 0.2
    w_bias_h2 = -0.5
    w_bias_h3 = 0.3
    w_bias_o = -0.1
    # /intial wieghts

    weight_list=[w_bias_h1,w_a_h1,w_b_h1,w_bias_h2,w_a_h2,w_b_h2,w_bias_h3,w_a_h3,w_b_h3,w_bias_o,w_h1_o,w_h2_o,w_h3_o]

    for i in range(0,11):
        print('-', end=" ")
    for i in range(0,len(weight_list)):
        if(i==len(weight_list)-1):
            print(round(weight_list[i], 5))
        else:
            print(round(weight_list[i],5),end=" ")


    i=0
    while i< int(iterations):
        for j in range(0,len(x_train)):
            # inputs
            bias=1
            a=x_train[j][0]
            b=x_train[j][1]

            summation_h1=bias*w_bias_h1 + a*w_a_h1 +b*w_b_h1
            # net summation value of h1
            output_h1=1 / (1 + np.exp(-summation_h1))
            # activation function

            summation_h2=bias*w_bias_h2 + a*w_a_h2 +b*w_b_h2
            output_h2=1 / (1 + np.exp(-summation_h2))

            summation_h3 = bias * w_bias_h3 + a * w_a_h3 + b * w_b_h3
            output_h3 = 1 / (1 + np.exp(-summation_h3))

            summation_o = output_h1*w_h1_o +output_h2*w_h2_o+output_h3*w_h3_o+ bias*w_bias_o
            output_final= 1 / (1 + np.exp(-summation_o))

            error=y_train[j]-output_final #target-actual

            #BACK PROPOGATION
            delta_o=output_final*(1-output_final)*(y_train[j]-output_final)
            delta_h1=output_h1*(1-output_h1)*(w_h1_o*delta_o)
            delta_h2=output_h2*(1-output_h2)*(w_h2_o*delta_o)
            delta_h3=output_h3*(1-output_h3)*(w_h3_o*delta_o)

            delta_w_h1_o= learning_rate*output_h1*delta_o
            delta_w_h2_o=learning_rate*output_h2*delta_o
            delta_w_h3_o=learning_rate*output_h3*delta_o
            delta_bias_o=learning_rate*1*delta_o

            w_h1_o=w_h1_o+delta_w_h1_o
            w_h2_o=w_h2_o+delta_w_h2_o
            w_h3_o=w_h3_o+delta_w_h3_o
            w_bias_o=w_bias_o+delta_bias_o

            delta_w_a_h1=learning_rate*a*delta_h1
            delta_w_b_h1=learning_rate*b*delta_h1
            delta_w_bias_h1=learning_rate*1*delta_h1

            w_a_h1=w_a_h1+delta_w_a_h1
            w_b_h1=w_b_h1+delta_w_b_h1
            w_bias_h1=w_bias_h1+delta_w_bias_h1

            delta_w_a_h2=learning_rate*a*delta_h2
            delta_w_b_h2=learning_rate*b*delta_h2
            delta_w_bias_h2=learning_rate*1*delta_h2

            w_a_h2=w_a_h2+delta_w_a_h2
            w_b_h2=w_b_h2+delta_w_b_h2
            w_bias_h2=w_bias_h2+delta_w_bias_h2

            delta_w_a_h3=learning_rate*a*delta_h3
            delta_w_b_h3=learning_rate*b*delta_h3
            delta_bias_h3=learning_rate*1*delta_h3

            w_a_h3=w_a_h3+delta_w_a_h3
            w_b_h3=w_b_h3+delta_w_b_h3
            w_bias_h3=w_bias_h3+delta_bias_h3

            new_list=[a,b,output_h1,output_h2,output_h3,output_final,int(y_train[j]),delta_h1,delta_h2,delta_h3,delta_o,w_bias_h1,
                      w_a_h1,w_b_h1,w_bias_h2,w_a_h2,w_b_h2,w_bias_h3,w_a_h3,w_b_h3,w_bias_o,w_h1_o,w_h2_o,w_h3_o]
            for k in range(0,len(new_list)):
                if(k==0 or k==1):
                    print(round(new_list[k],5),end=" ")
                elif(k==6):
                    print(new_list[k],end=" ")
                elif(k==len(new_list)-1):
                    print(round(new_list[k], 5))
                else:
                    print(round(new_list[k], 5), end=" ")


        i+=1

if __name__ == "__main__":
    expected_args = ["--data","--eta","--iterations"]
    arg_len = len(sys.argv)
    info = []

    for i in range(len(expected_args)):
        for j in range(1, len(sys.argv)):
            if expected_args[i] == sys.argv[j] and sys.argv[j + 1]:
                info.append(sys.argv[j + 1])

    data = pd.read_csv(info[0], header=None)

    learning_rate=info[1]
    iterations=info[2]
    x_train = data.iloc[:,:-1].values
    y_train = data.iloc[:, -1].values
    # print(x_train, type(x_train))

    create_nueral_network(x_train,y_train,float(learning_rate),iterations)