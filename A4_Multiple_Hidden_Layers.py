import numpy as np


def sigmoid(x):
    return 1/(1+2.718**(-x))


def feed_forward(x1, x2, W1, B1, W2, B2, W3, B3):
    # feed forward
    # first we will calculte the output of hidden layer 1 neurons
    # first neuron
    b1in = x1*W1[0][0]+x2*W1[1][0]+B1[0][0]
    b1out = sigmoid(b1in)
    # second neuron
    b2in = x1*W1[0][1]+x2*W1[1][1]+B1[1][0]
    b2out = sigmoid(b2in)
    # third neuron
    b3in = x1*W1[0][2]+x2*W1[1][2]+B1[2][0]
    b3out = sigmoid(b3in)
    # output of hidden layer 1
    hiddenoutputs = [b1out, b2out, b3out]
    # now we will calculate the output of hidden layer 2 neurons
    # first neuron
    b4in = b1out*W2[0][0]+b2out*W2[1][0]+b3out*W2[2][0]+B2[0][0]
    b4out = sigmoid(b4in)
    # second neuron
    b5in = b1out*W2[0][1]+b2out*W2[1][1]+b3out*W2[2][1]+B2[1][0]
    b5out = sigmoid(b5in)
    # output of hidden layer 2
    hiddenoutputs2 = [b4out, b5out]
    # now we will calculate the output at output layer
    b6in = b4out*W3[0][0]+b5out*W3[1][0]+B3[0][0]
    output = sigmoid(b6in)
    return hiddenoutputs, hiddenoutputs2, output


def back_propagation(x1, x2, y, W1, B1, W2, B2, hiddenoutputs, output, eta, W3, B3, hiddenoutputs2):
    ErrorOutput = output*(1-output)*(y-output)
    # calculating error at hidden layer 2
    Errorb4 = hiddenoutputs2[0]*(1-hiddenoutputs2[0])*(W3[0][0]*ErrorOutput)
    Errorb5 = hiddenoutputs2[1]*(1-hiddenoutputs2[1])*(W3[1][0]*ErrorOutput)
    # calculating error at hidden layer 1
    Errorb1 = hiddenoutputs[0]*(1-hiddenoutputs[0]) * \
        (W2[0][0]*Errorb4+W2[0][1]*Errorb5)
    Errorb2 = hiddenoutputs[1]*(1-hiddenoutputs[1]) * \
        (W2[1][0]*Errorb4+W2[1][1]*Errorb5)
    Errorb3 = hiddenoutputs[2]*(1-hiddenoutputs[2]) * \
        (W2[2][0]*Errorb4+W2[2][1]*Errorb5)

    # updating weights and biases
    # updating weights and biases of output layer
    W3[0][0] = W3[0][0]+eta*ErrorOutput*hiddenoutputs2[0]
    W3[1][0] = W3[1][0]+eta*ErrorOutput*hiddenoutputs2[1]
    B3[0][0] = B3[0][0]+eta*ErrorOutput

    # updating weights and biases of hidden layer 2
    W2[0][0] = W2[0][0]+eta*Errorb4*hiddenoutputs[0]
    W2[1][0] = W2[1][0]+eta*Errorb4*hiddenoutputs[1]
    W2[2][0] = W2[2][0]+eta*Errorb4*hiddenoutputs[2]
    W2[0][1] = W2[0][1]+eta*Errorb5*hiddenoutputs[0]
    W2[1][1] = W2[1][1]+eta*Errorb5*hiddenoutputs[1]
    W2[2][1] = W2[2][1]+eta*Errorb5*hiddenoutputs[2]
    B2[0][0] = B2[0][0]+eta*Errorb4
    B2[1][0] = B2[1][0]+eta*Errorb5

    # updating weights and biases of hidden layer 1

    W1[0][0] = W1[0][0]+eta*Errorb1*x1
    W1[1][0] = W1[1][0]+eta*Errorb1*x2
    W1[0][1] = W1[0][1]+eta*Errorb2*x1
    W1[1][1] = W1[1][1]+eta*Errorb2*x2
    W1[0][2] = W1[0][2]+eta*Errorb3*x1
    W1[1][2] = W1[1][2]+eta*Errorb3*x2
    B1[0][0] = B1[0][0]+eta*Errorb1
    B1[1][0] = B1[1][0]+eta*Errorb2
    B1[2][0] = B1[2][0]+eta*Errorb3

    return W1, B1, W2, B2, W3, B3


def main():

    # input data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # output data
    y = np.array([[0], [1], [1], [0]])
    # initializing weights and biases randomly
    W1 = np.random.rand(2, 3)
    B1 = np.random.rand(3, 1)
    W2 = np.random.rand(3, 2)
    B2 = np.random.rand(2, 1)
    W3 = np.random.rand(2, 1)
    B3 = np.random.rand(1, 1)

    # learning rate(randomly chosen)
    eta = 0.1
    print('Initial Weights: ', W1, W2, W3)
    print('Initial Biases: ', B1, B2, B3)
    print('***********************************')
    for i in range(4):
        print("Input: ", X[i])
        print("Target: ", y[i])
        hiddenoutputs, hiddenoutputs2, output = feed_forward(
            X[i][0], X[i][1], W1, B1, W2, B2, W3, B3)
        W1, B1, W2, B2, W3, B3 = back_propagation(
            X[i][0], X[i][1], y[i], W1, B1, W2, B2, hiddenoutputs, output, eta, W3, B3, hiddenoutputs2)
        print("Iteration: ", i+1)
        print("Output: ", output)
        print("Weights: ", W1, W2, W3)
        print("Biases: ", B1, B2, B3)
        print("")


if __name__ == "__main__":
    main()
