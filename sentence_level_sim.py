import gensim
#load word2vec model, here GoogleNews is used
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
#two sample sentences 
s1 = 'the first sentence'
s2 = 'the second text'

#calculate distance between two sentences using WMD algorithm
distance = model.wmdistance(s1, s2)

print ('distance = %.3f' % distance)
