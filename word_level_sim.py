import math
import numpy as np

std_embeddings_index = {}
with open('numberbatch-en-17.06.txt') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        std_embeddings_index[word] = embedding

def cosineValue(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)


def get_sentence_vector(sentence, std_embeddings_index = std_embeddings_index ):
    sent_vector = 0
    for word in sentence.lower().split():
        if word not in std_embeddings_index :
            word_vector = np.array(np.random.uniform(-1.0, 1.0, 300))
            std_embeddings_index[word] = word_vector
        else:
            word_vector = std_embeddings_index[word]
        sent_vector = sent_vector + word_vector

    return sent_vector

def cosine_sim(sent1, sent2):
    return cosineValue(get_sentence_vector(sent1), get_sentence_vector(sent2))

s1 = "This is a foo bar sentence ."
s2 = "This sentence is similar to a foo bar sentence ."
s3 = "What is this string ? Totally not related to the other two lines ."

print cosine_sim(s1, s2) # Should give high cosine similarity
print cosine_sim(s1, s3) # Shouldn't give high cosine similarity value
print cosine_sim(s2, s3) # Shouldn't give high cosine similarity value

