from gensim import utils
import gensim
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


model = Doc2Vec.load('doc2vec.model')
#print(model.docvecs.most_similar(9))

train_arrays = numpy.zeros((50000, 100))
train_labels = numpy.zeros(50000)

test_arrays=numpy.zeros((2000,100))
test_labels=numpy.zeros(2000)





for i in range(25001):
    train_arrays[i] = model.docvecs[i]
    train_labels[i] = 1

j=0
for i in range(25001,26000):
	test_arrays[j]=model.docvecs[i]
	test_labels[j]=1
	j=j+1


for i in range(25001,50000):
	train_arrays[i]=model.docvecs[i]
	train_labels[i]=0

#reinitialize
j=1001
for i in range(50001,51000):
	test_arrays[j]=model.docvecs[i]
	test_labels[j]=0
	j=j+1






classifier=MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
classifier.fit(train_arrays,train_labels)
print(classifier.score(test_arrays,test_labels))


# classifier=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#           intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
# classifier.fit(train_arrays,train_labels)

# for i in range(1000):
# 	print(classifier.predict(train_arrays[i]))
# 	print("\n")







#print(train_labels)

# docvec = d2v_model.docvecs[1]
# print(docvec)


# similar_doc=d2v_model.docvecs.most_similar(14)
# print(similar_doc)
    	






