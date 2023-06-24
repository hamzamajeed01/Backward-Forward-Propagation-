import numpy as np


def sigmoid(x):
    return 1/(1+2.718**(-x))


def feed_forward(x1, x2, W1, B1, W2, B2):
    # feed forward
    # first we will calculte the output of hidden layer neurons
    # first neuron
    b1in = x1*W1[0][0]+x2*W1[1][0]+B1[0][0]
    b1out = sigmoid(b1in)
    # second neuron
    b2in = x1*W1[0][1]+x2*W1[1][1]+B1[1][0]
    b2out = sigmoid(b2in)
    # third neuron
    b3in = x1*W1[0][2]+x2*W1[1][2]+B1[2][0]
    b3out = sigmoid(b3in)
    # store all 3 outputs in a list
    hiddenoutputs = [b1out, b2out, b3out]
    # output at output layer
    bout = b1out*W2[0][0]+b2out*W2[1][0]+b3out*W2[2][0]+B2[0][0]
    output = sigmoid(bout)
    return hiddenoutputs, output


def back_propagation(x1, x2, y, W1, B1, W2, B2, hiddenoutputs, output, eta):
    ErrorOutput = output*(1-output)*(y-output)
    Errorb1 = hiddenoutputs[0]*(1-hiddenoutputs[0])*(W2[0][0]*ErrorOutput)
    Errorb2 = hiddenoutputs[1]*(1-hiddenoutputs[1])*(W2[1][0]*ErrorOutput)
    Errorb3 = hiddenoutputs[2]*(1-hiddenoutputs[2])*(W2[2][0]*ErrorOutput)
    # updating weights and biases
    # updating weights and biases of output layer
    W2[0][0] = W2[0][0]+eta*ErrorOutput*hiddenoutputs[0]
    W2[1][0] = W2[1][0]+eta*ErrorOutput*hiddenoutputs[1]
    W2[2][0] = W2[2][0]+eta*ErrorOutput*hiddenoutputs[2]
    B2[0][0] = B2[0][0]+eta*ErrorOutput
    # updating weights and biases of hidden layer
    W1[0][0] = W1[0][0]+eta*Errorb1*x1
    W1[1][0] = W1[1][0]+eta*Errorb1*x2
    B1[0][0] = B1[0][0]+eta*Errorb1

    W1[0][1] = W1[0][1]+eta*Errorb2*x1
    W1[1][1] = W1[1][1]+eta*Errorb2*x2
    B1[1][0] = B1[1][0]+eta*Errorb2

    W1[0][2] = W1[0][2]+eta*Errorb3*x1
    W1[1][2] = W1[1][2]+eta*Errorb3*x2
    B1[2][0] = B1[2][0]+eta*Errorb3
    return W1, B1, W2, B2


def main():

    # input data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # output data
    y = np.array([[0], [1], [1], [0]])
    # initializing weights and biases randomly
    W1 = np.random.randn(2, 3)
    B1 = np.random.randn(3, 1)
    W2 = np.random.randn(3, 1)
    B2 = np.random.randn(1, 1)
    # learning rate(randomly chosen)
    eta = 0.1
    print('Initial Weights: ', W1, W2)
    print('Initial Biases: ', B1, B2)
    print('***********************************')
    for i in range(4):
        print("Input: ", X[i])
        hiddenoutputs, output = feed_forward(X[i][0], X[i][1], W1, B1, W2, B2)
        W1, B1, W2, B2 = back_propagation(
            X[i][0], X[i][1], y[i][0], W1, B1, W2, B2, hiddenoutputs, output, eta)
        print("Iteration: ", i+1)
        print("Output: ", output)
        print("Weights: ", W1, W2)
        print("Biases: ", B1, B2)
        print("")


if __name__ == "__main__":
    main()
