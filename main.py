import math

with open('ps5_data.csv') as d:
    data = d.readlines()
    # Reshape loaded data to 5000x400 and convert to float
    for i in range(len(data)):
        data[i] = data[i].split(',')
        for j in range(len((data[i]))):
            data[i][j] = float(data[i][j])

with open('ps5_data-labels.csv') as l:
    labels = l.readlines()
    # Reshape list with 5000 elements
    for i in range(len(labels)):
        labels[i] = int(labels[i].strip('\n'))


with open('ps5_theta1.csv') as t1:
    theta1 = t1.readlines()
    # Reshape to 25x401 list of lists
    for i in range(len(theta1)):
        theta1[i] = theta1[i].split(',')
        for j in range(len((theta1[i]))):
            theta1[i][j] = float(theta1[i][j])

with open('ps5_theta2.csv') as t2:
    theta2 = t2.readlines()
    # Reshape to 10x26 list of lists
    for i in range(len(theta2)):
        theta2[i] = theta2[i].split(',')
        for j in range(len((theta2[i]))):
            theta2[i][j] = float(theta2[i][j])


# First element of each row is bias
bias1 = theta1[0]
bias2 = theta2[0]
theta1 = theta1[1:]  # 25x400
theta2 = theta2[1:]  # 10x25


def list_reshape(d, row_num, col_num):
    # Iterate through original list
    mat = []
    for r in range(row_num):
        row = []
        for c in range(col_num):
            row.append(data[row_num * r + c])
        mat.append(row)
    return mat


def neuron(x, weights, bias):
    z = 0
    z += bias
    for i in range(len(x)):
        z += (x[i] * weights[i])
    try:
        e = math.exp(-z)
    except OverflowError:
        e = float('inf')
    return 1/(1 + e)


def softmax(vector):
    length = len(vector)
    ak = []
    total = 0
    for i in range(length):
        num = math.exp(vector[i])
        ak.append(num)
        total += num
    for j in range(length):
        ak[j] /= total
    return ak


# Input(25), Weight(10x25), bias(10)
def last_layer(input, weight, bias):
    layer = []
    # Iterate over 10
    for i in range(len(weight)):
        layer.append(neuron(input, weight[i], bias[i]))
    return softmax(layer)


def forward_propagation(x):
    global theta1, theta2, bias1, bias2
    layer = []

    # From input layer to hidden layer
    for i in range(len(theta1)):
        n = neuron(x, theta1[i], bias1[i])
        layer.append(n)
    """
    # From hidden layer to softmax layer
    for j in range(len(theta2)):
        n = neuron(layer1, theta2[j], bias2)
        layer2.append(n)
    """
    return last_layer(layer, theta2, bias2)


# Classify an image as digit 0~9
def digit_classifier(x):
    return forward_propagation(x).index(max(forward_propagation(x)))


predicted = []

# Classify all 5000 images and return error rate
def classification():
    global data, labels, predicted
    error = 0
    for i in range(len(data)):
        print('predict: ', digit_classifier(data[i]), 'result: ', labels[i])
        prediction = forward_propagation(data[i])
        # index and prediction digit are shifted by one
        predicted.append(prediction)
        if (digit_classifier(prediction)) != labels[i]:
            error += 1
    return (error/len(labels))
# error rate: 87.54%


# MLE Cost Function
def cost_function(predicted, labels):
    cost = 0
    for i in range(len(predicted)):
        for j in range(9):
            cost += (float(labels[i]) * float(math.log(predicted[i][j]))) + ((1.0 - float(labels[i]))
            * (1.0 - float(math.log(predicted[i][j]))))

    cost = float(-cost/len(labels))

    return cost


# Pseudo-code for backward propagation
"""
def back_propagation(data):
    global predicted, labels
    set delta2 matrix to 0
    for i in range(len(data)):
        set a = data[i]
        forward_propagation for a(l) for l = 2 and 3(=l)
        delta2 = predicted[1][i] - labels[i]
    w10 = index 10 of Transpose(predicted[0]).dot_product(delta2)
"""

if __name__ == "__main__":
    print('last layer', classification())
    print('predicted', predicted)
    print('cost function', cost_function(predicted, labels))
