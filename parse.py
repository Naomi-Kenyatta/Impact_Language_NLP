import numpy as np
#from IPython.display import display, HTML
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import re
import csv


def clean_file_data(file_contents,stop_word_removal='no'):
    '''Input a list of a file. Output a list with every sentance parsed.
    Step 1 in the process. Parsing the file means removing everything
    that is not a letter and to remove stop words'''
    text = []
    for val in file_contents.split('.'):
        sent = re.findall("[A-Za-z]+", val)
        line = ''
        for words in sent:
            if stop_word_removal == 'yes':
                #check if it is in stopwords
                if len(words) > 1 and words not in stop_words:
                    line = line + ' ' + words
            else:
                if len(words) > 1 :
                    line = line + ' ' + words
        text.append(line)
    return text


def generate_dictionary_data(text):
    word_to_index= dict()
    index_to_word = dict()
    corpus = []
    count = 0
    vocab_size = 0

    for row in text:
        for word in row.split():
            word = word.lower()
            corpus.append(word)
            if word_to_index.get(word) == None:
                word_to_index.update ( {word : count})
                index_to_word.update ( {count : word })
                count  += 1
    vocab_size = len(word_to_index)
    length_of_corpus = len(corpus)

    return word_to_index,index_to_word,corpus,vocab_size,length_of_corpus


def get_one_hot_vectors(target_word,context_words,vocab_size,word_to_index):

    #Create an array of size = vocab_size filled with zeros
    trgt_word_vector = np.zeros(vocab_size)

    #Get the index of the target_word according to the dictionary word_to_index.
    #If target_word = best, the index according to the dictionary word_to_index is 0.
    #So the one hot vector will be [1, 0, 0, 0, 0, 0, 0, 0, 0]
    index_of_word_dictionary = word_to_index.get(target_word)

    #Set the index to 1
    trgt_word_vector[index_of_word_dictionary] = 1

    #Repeat same steps for context_words but in a loop
    ctxt_word_vector = np.zeros(vocab_size)


    for word in context_words:
        index_of_word_dictionary = word_to_index.get(word)
        ctxt_word_vector[index_of_word_dictionary] = 1

    return trgt_word_vector,ctxt_word_vector



def generate_training_data(corpus,window_size,vocab_size,word_to_index,length_of_corpus,sample=None):

    training_data =  []
    training_sample_words =  []
    for i,word in enumerate(corpus):

        index_target_word = i
        target_word = word
        context_words = []

        #when target word is the first word
        if i == 0:

            # trgt_word_index:(0), ctxt_word_index:(1,2)
            context_words = [corpus[x] for x in range(i + 1 , window_size + 1)]


        #when target word is the last word
        elif i == len(corpus)-1:

            # trgt_word_index:(9), ctxt_word_index:(8,7), length_of_corpus = 10
            context_words = [corpus[x] for x in range(length_of_corpus - 2 ,length_of_corpus -2 - window_size  , -1 )]

        #When target word is the middle word
        else:

            #Before the middle target word
            before_target_word_index = index_target_word - 1
            for x in range(before_target_word_index, before_target_word_index - window_size , -1):
                if x >=0:
                    context_words.extend([corpus[x]])

            #After the middle target word
            after_target_word_index = index_target_word + 1
            for x in range(after_target_word_index, after_target_word_index + window_size):
                if x < len(corpus):
                    context_words.extend([corpus[x]])


        trgt_word_vector,ctxt_word_vector = get_one_hot_vectors(target_word,context_words,vocab_size,word_to_index)
        training_data.append([trgt_word_vector,ctxt_word_vector])

        if sample is not None:
            training_sample_words.append([target_word,context_words])

    return training_data,training_sample_words


if __name__ == "__main__":
    #getting the file data. Will later need to go file by file. Everything is lower case
    file_contents = []
    with open('data/raw/documents.txt') as f:
        file_contents = f.read().lower()

    stop_words = []
    #get the stop words
    with open('resources/StopWords_Generic.txt') as f:
        stop_words = [line.rstrip('\n') for line in f]
        stop_words = [x.lower() for x in stop_words]

    text = clean_file_data(file_contents,"yes")

    word_to_index,index_to_word,corpus,vocab_size,length_of_corpus = generate_dictionary_data(text)
    print('Number of unique words:' , vocab_size)
    print('Length of corpus :',length_of_corpus)

    window_size = 2 #how large in the surrondings do we want to consider
    training_data,training_sample_words = generate_training_data(corpus,window_size,vocab_size,word_to_index,length_of_corpus,'yes')
    #save the training data
    with open('data/processed/take1.txt', 'w') as fp:
        for i in training_data:
            data = ""
            for j in i:
                #print([str(elem) for m,elem in enumerate(list(j)))])
                data +=  ' '.join([str(elem) for m,elem in enumerate(list(j))]) + ","
            #data = re.sub(r'[^\w\s]', '', data)

            data += "\n"
            fp.write(data)

    # Save dict
    w = csv.writer(open("data/processed/dict.csv", "w"))
    for key, val in index_to_word.items():
        w.writerow([key, val])
