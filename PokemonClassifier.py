# Raghav Gupta
# AI/Mixed Reality Lab
# Next Tech Lab

import pandas as pd
import numpy as np
from TheanoNN import NN
import theano.tensor as T
import theano
# from sklearn.ensemble import ExtraTreesClassifier

rng = np.random.RandomState(1234)

f = open("Pokemon.csv")
dataset = pd.read_csv(f)
# Below commented lines are for randomising the order of rows for split b/w training and test data
# dataset = dataset.sample(frac=1).reset_index(drop=True)
# pd.DataFrame.to_csv(dataset,"Pokemon.csv")
f.close()

labels = dataset["Type 1"]
# Dropped Generations column as it overfits the data rather than contributing much as a feature (see below for etc.feature_importances_)
# Dropped Total column as it is anyways calculated using the other features, hence would overfit
X = dataset[["HP","Attack","Defense","Sp. Atk","Sp. Def","Speed"]]
distinctLabels = np.array(labels.unique())
# Each Type Label is given an integer and mapping is stored in a dictionary label2int; the inverse dictionary is int2label
label2int = dict((l,i) for (i,l) in enumerate(distinctLabels))
int2label = dict((i,l) for (i,l) in enumerate(distinctLabels))
print label2int
l = []
for i in labels:
    l.append(label2int[i])

labels = np.array(l).astype(dtype=np.uint8)

# etc = ExtraTreesClassifier()
# etc.fit(X,labels)
# print etc.feature_importances_
#     HP           Attack      Defense    Spl. Atk   Spl. Def    Speed      Generation
# [ 0.15094212  0.15045182  0.14142085  0.16435534  0.14503864  0.15237957  0.09541167 ]

ndata = X.shape[0]
# Split train-test data in the ratio 70:30
ntrain = int(70 * ndata / 100)
trainX, trainY = X[:ntrain], labels[:ntrain]
testX, testY = X[ntrain:], labels[ntrain:]

# inputs and labels are stored as theano shared variables
def prep_dataset(X, labels):
    x = theano.shared(np.asarray(X, dtype=theano.config.floatX), borrow=True)
    y = theano.shared(np.asarray(labels, dtype=theano.config.floatX), borrow=True)
    return x, T.cast(y, 'int32')

trainX, trainY = prep_dataset(trainX,trainY)
testX, testY = prep_dataset(testX,testY)

x = T.matrix('x')
y = T.ivector('y')

pi = T.matrix('pi')

# A neural network with a single hidden layer with 130 nodes
nn = NN(input=x, rng=rng, n_in=trainX.get_value(borrow=True).shape[1],n_hidden1=130,n_out=18)

a = 0.01 # L1 regularisation parameter
b = 0.001 # L2 regularisation parameter
alpha = 0.0005 # Learning rate

cost = (nn.negLogLik(y)+a*nn.L1+b*nn.L2)

gparams = [T.grad(cost,param) for param in nn.params]
train_model = theano.function(inputs=[],outputs=cost,updates=[(param,param-alpha*gparam) for param,gparam in zip(nn.params,gparams)],givens={x:trainX,y:trainY})
test_model = theano.function(inputs=[],outputs=nn.error(y),givens={x:testX,y:testY})
testTraining = theano.function(inputs=[],outputs=nn.error(y),givens={x:trainX,y:trainY})
predict_model = theano.function(inputs=[pi],outputs=nn.outl.prediction,givens={x:pi})

n_epochs = 15000
lowest_error = 1

for i in range(n_epochs):
    print "\n Epoch : ",i,"/",n_epochs
    trainCost = train_model()
    testError = test_model()
    print " Test Error : ",testError
    trainError = testTraining()
    print " Train Error : ",trainError
    if lowest_error > testError:
        lowest_error = testError
        bestEpoch = i

print "\n\n Accuracy : ",100-testError*100,"%"
print " Best Accuracy at Epoch ",bestEpoch," : ",100-lowest_error*100,"%"

# Predicting user-input stats of Pokemon
print "\n"
ch = raw_input("Predict class of a Pokemon? Y/N : ")
while ch=='Y' or ch=='y':
    stats = []
    v = input(" Enter HP : ")
    stats.append(v)
    v = input(" Enter Attack : ")
    stats.append(v)
    v = input(" Enter Defense : ")
    stats.append(v)
    v = input(" Enter Spl. Atk : ")
    stats.append(v)
    v = input(" Enter Spl. Def : ")
    stats.append(v)
    v = input(" Enter Speed : ")
    stats.append(v)
    pr = predict_model([stats,[0,0,0,0,0,0]])
    print " Prediction : ",int2label[pr[0]]," Type; at ",100-testError*100,"% accuracy"
    print"\n"
    ch = raw_input(" Predict another pokemon? Y/N : ")
