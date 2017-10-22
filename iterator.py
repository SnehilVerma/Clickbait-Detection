import gensim
import csv


#preprocessing to create list of lists to feed to word2vec model.
data=[]
clickbait=[]
with open('clickbait_data.txt') as fp:
	data=fp.read().split("\n")
	

for item in data:
	row=item.split()
	clickbait.append(row)




for item in clickbait:
	if not item:
		clickbait.remove(item)

clickbait=clickbait[:-2]
#print(clickbait)
#CLICKBAIT SET



non_clickbait=[]
with open('non_clickbait_data.txt') as fp:
	data=fp.read().split("\n")
	

for item in data:
	row=item.split()
	non_clickbait.append(row)




for item in non_clickbait:
	if not item:
		non_clickbait.remove(item)


print(non_clickbait)
#NONCLICKBAIT SET



cb_model=gensim.models.Word2Vec(clickbait,min_count=1,size=300,workers=4)
ncb_model=gensim.models.Word2Vec(non_clickbait,min_count=1,size=300,workers=4)

sentence=['32','Amazing','DIY','Costumes','That','Prove','Halloween','Is','Actually','Meant','For','Teens']
score_cb=0;
score_ncb=0;
for item in sentence:
	print(cb_model[item])
	#score_ncb=score_ncb+ncb_model[item]	


#print(score_cb + 'CB')
#print(score_ncb+ 'NCB')



	





