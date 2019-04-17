#!/usr/bin/env python

from math import exp, log


def logistic_fn(x):
    try:
        return 1.0 / (1.0 + exp(-x))
    except:
        if x < 709.:
            return 0
        print(x)
        raise OverflowError


def logistic_fn_prime(x):
    return logistic_fn(x) * (1.0 - logistic_fn(x))


def logistic(x, beta):
    return logistic_fn(sum(ix*ibeta for ix, ibeta in zip([1.]+x, beta)))

# df(x .b)/db_i = df(x.b) / d(x.b) * d(x.b) / db_i = df(x.b) / d(x.b) * x_i = f(x.b) * (1 - f(x.b)) * x_i
def logistic_prime_j(x, beta, j):
    return logistic(x, beta) * (1.0 - logistic(x, beta)) * ([1] + x)[j]

def logistic_prime(x, beta):
    return [logistic_prime_j(x, beta, j) for j in range(len(beta))]

import random
def in_random_order(data):
    indices = [i for i, _ in enumerate(data)]
    random.shuffle(indices)
    for i in indices:
        yield data[i]

def stochastic_minimize(target_fn, gradient_fn, x, y,
                        theta_0, alpha_0: float = 0.01, iterations=20):
    data = list(zip(x, y))
    theta = theta_0
    alpha = alpha_0
    min_theta, min_value = None, float("inf")
    iterations_with_no_improvement = 0
    iter = 0
    while iterations_with_no_improvement < iterations:
        value = sum(target_fn(x_i, y_i, theta) for x_i, y_i in data)
        if iter % 100 == 0: print(iter, value, theta)
        iter += 1
        if value < min_value:
            min_theta, min_value = theta, value
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            iterations_with_no_improvement += 1
            alpha *= 0.9
            theta = min_theta
        for x_i, y_i in in_random_order(data):
            gradient_i = list(gradient_fn(x_i, y_i, theta))
            theta = list(t - alpha*g for t, g in zip(theta, gradient_i))
    return min_theta

def logistic_log_likelihood(x, y, beta):
    l = logistic(x, beta)
    try:
        if y < 0.5:
            if l == 1:
                return log(1e-100)
            return log(1-l)
        else:
            if l == 0: l = 1e-100
            return log(l)
    except:
        print("VE, LLL", x, y, beta, l)
        raise ValueError

# F(x, y, b) = y*log(f(x, b)) + (1-y)*log(1 - f(x, b))
# F'(x, y, b) = y*f'(x,b)/f(x, b) + (1-y)*(-f'(x,b))/(1 - f(x, b))
def logistic_log_likelihood_prime(x, y, beta):
    l = logistic(x, beta)
    lp = [logistic_prime_j(x, beta, j) for j in range(len(beta))]
    try:
        if y < 0.5:
            if l == 1.: l = 0.9999999999
            return [-lpj/(1-l) for lpj in lp]
        else:
            if l == 0.: l = 0.0000000001
            return [lpj/l for lpj in lp]
    except:
        print("VE, LLLP", x, y, b, l, lp)
        raise ValueError

logistic_log_likelihood_gradient = logistic_log_likelihood_prime

def logistic_regression_sgd(x0, x1, beta0, alpha0=0.01, iterations=20):
    x = x0 + x1
    y = [0]*len(x0) + [1]*len(x1)
    return stochastic_minimize(lambda x, y, b: -logistic_log_likelihood(x, y, b) + sum(ib*ib for ib in b),
                               # F(x, y, b) = (f(x, b) -y)^2. dF(x,y,b)/db_j = 2(f(x,b)-y)df(x,b)/db_j
                               lambda x, y, b: [-l + 2*ib for l, ib in zip(logistic_log_likelihood_prime(x, y, b), b)],
                               x, y, beta0, alpha0, iterations)

if __name__ == "__main__":
    import csv
    diabetes = csv.reader(open('diabetes.csv'))
    header = diabetes.__next__()  # change to diabetes.next() for python2!
    data_ = list(d for d in diabetes)
    data = [[float(di) for di in d[:-1]] for d in data_]
    target = [int(d[-1]) for d in data_]
    # print("Header:", header)
    x0 = [d for d, t in zip(data, target) if t == 0]
    x1 = [d for d, t in zip(data, target) if t == 1]

def accuracy(data, target, f):
    right = 0
    tot = 0
    for d, t in zip(data, target):
        if int(f(d) > 0.5) == t:
            right += 1
        tot += 1
    return right, tot, right / float(tot)
