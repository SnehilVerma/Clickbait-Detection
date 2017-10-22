from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import gensim
import numpy
from random import shuffle




# data=[]
# clickbait=[]
# non_clickbait=[]
# with open('clickbait_data.txt') as fp:
# 	data=fp.read().split("\n")

# for item in data:
# 	item=item.lower()
# 	row=item.split()
# 	clickbait.append(row)

# for item in clickbait:
# 	if not item:
# 		clickbait.remove(item)

# clickbait=clickbait[:-100]

# #setting tags in labeled sentence.
# for item in clickbait:
# 	x=["CLICKBAIT"]
# 	item.append(x)





# #NON CLICK BAIT
# with open('non_clickbait_data.txt') as fp:
# 	data=fp.read().split("\n")
	
# for item in data:
# 	item=item.lower()
# 	row=item.split()
# 	non_clickbait.append(row)

# for item in non_clickbait:
# 	if not item:
# 		non_clickbait.remove(item)

# non_clickbait=non_clickbait[:-100]

# for item in non_clickbait:
# 	x=["NON_CLICKBAIT"]
# 	item.append(x)

# print(non_clickbait)



data=[]
clickbait=[]
click_labels=[]
non_clickbait=[]
#non_click_labels=[]
with open('clickbait_data.txt') as fp:
  data=fp.read().split("\n")

i=0
for item in data:
  click_labels.append('TRAIN_CLICK_'+str(i))
  item=item.lower()
  clickbait.append(item)
  i=i+1


print(i)

#NON CLICK BAIT
with open('non_clickbait_data.txt') as fp:
  data=fp.read().split("\n")
    
for item in data:
    click_labels.append('TRAIN_NON_CLICK'+str(i))
    item=item.lower()
    clickbait.append(item)
    i=i+1

print(i)


class LabeledLineSentence(object):

    def __init__(self, doc_list, labels_list):

        self.labels_list = labels_list
        self.doc_list = doc_list

    def __iter__(self):

        for idx, doc in enumerate(self.doc_list):
              yield gensim.models.doc2vec.LabeledSentence(doc,    
[self.labels_list[idx]])


it=LabeledLineSentence(clickbait,click_labels)
model = Doc2Vec(min_count=1,size=100,window=10,workers=8,alpha=0.025, min_alpha=0.025)
model.build_vocab(it)



            
for epoch in range(10):
 print('iteration'+str(epoch+1))
 model.train(it,total_examples=model.corpus_count,epochs=model.iter)
 model.alpha -= 0.002
 model.min_alpha = model.alpha
 model.train(it,total_examples=model.corpus_count,epochs=model.iter)


model.save('doc2vec.model')
print("saved model")