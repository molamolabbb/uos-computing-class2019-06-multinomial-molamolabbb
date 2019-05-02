#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python

from math import exp
# Probably you won't need all of these
from logistic import (logistic_regression_sgd, stochastic_minimize, logistic, accuracy, logistic_log_likelihood, logistic_log_likelihood_prime)


# In[2]:


def logistic_ridge_regression_sgd(x0,x1,alpha0,kappa,iterations):
    x = x0 + x1
    y = [0]*len(x0) + [1]*len(x1)
    return stochastic_minimize(lambda x, y, b: -logistic_log_likelihood(x, y, b) + sum(kappa*ib*ib for ib in b),
                               # F(x, y, b) = (f(x, b) -y)^2. dF(x,y,b)/db_j = 2(f(x,b)-y)df(x,b)/db_j
                               lambda x, y, b: [-l + 2*kappa*ib for l, ib in zip(logistic_log_likelihood_prime(x, y, b), b)],
                               x, y, beta0, alpha0, iterations)
def inner(x, beta):
    return beta[0]+sum(x[i]*beta[i+1] for i in range(len(x)))

def multi_logistic(x, betas):
    ex_sum = sum([exp(inner(x,betas[i])) for i in range(len(betas))])
    k = [exp(inner(x,betas[j]))/(ex_sum) for j in range(len(betas))]
    return k
def multi_best(x):
    best = []
    for i in range(len(x)):
        if max(x)==x[i]: best.append(1)
        else: best.append(0)
    return best

def multi_accuracy(x,y,betas):
    corr = 0
    for x_,y_ in zip(x,y):
        target = [0,0,0]
        target[y_]=1
        if target == multi_best(multi_logistic(x_,betas)):
            corr += 1
    print(corr/len(x))
    return corr/len(x)


# In[10]:


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
    beta0 = [0.01,0.01,0.01,0.01,0.01]
    
    results_02 = open('result_02.txt','w')
    results_12 = open('result_12.txt','w')
    results_20 = open('result_20.txt','w')
    results_10 = open('result_10.txt','w')
    multi = open("multi.txt",'w')
    
    beta02 = logistic_ridge_regression_sgd(x2,x0,alpha0=0.01,kappa=0.1,iterations=20)
    beta12 = logistic_ridge_regression_sgd(x2,x1,alpha0=0.01,kappa=0.1,iterations=20)
    beta20 = logistic_ridge_regression_sgd(x0,x2,alpha0=0.01,kappa=0.1,iterations=20)
    beta10 = logistic_ridge_regression_sgd(x0,x1,alpha0=0.01,kappa=0.1,iterations=20)
    multi_ref_2 = multi_accuracy(data,target,[beta02,beta12,[0]*len(beta02)])
    multi_ref_0 = multi_accuracy(data,target,[[0]*len(beta20),beta20,beta10])
    
    for i in range(len(beta02)):
        results_02.write('%f \n' %beta02[i])
        results_12.write('%f \n' %beta12[i])
        results_20.write('%f \n' %beta20[i])
        results_10.write('%f \n' %beta10[i])
    multi.write('reference[0] : %f \n reference[2] : %f' %(multi_ref_0,multi_ref_2))
    results_02.close()
    results_12.close()
    results_20.close()
    results_10.close()
    multi.close()


# In[ ]:




