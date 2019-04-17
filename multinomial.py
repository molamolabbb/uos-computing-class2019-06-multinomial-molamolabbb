#!/usr/bin/env python

from math import exp
# Probably you won't need all of these
from logistic import (logistic_regression_sgd, stochastic_minimize, logistic, accuracy, logistic_log_likelihood, logistic_log_likelihood_prime)

# As always, functions go here

if __name__ == "__main__":
    import csv
    iris = csv.reader(open('Fisher.txt'), delimiter='\t')
    header = iris.__next__()  # change to iris.next() for python2!
    data_ = list(d for d in iris)
    data = [[float(di) for di in d[1:]] for d in data_]
    target = [int(d[0]) for d in data_]
    print("Header:", header)
    # Get the data for the different categories
    x0 = [d for d, t in zip(data, target) if t == 0]
    x1 = [d for d, t in zip(data, target) if t == 1]
    x2 = [d for d, t in zip(data, target) if t == 2]
