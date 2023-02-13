import numpy as np
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import re
from model_functions import *
from plots import *
import pandas as pd


def read_files(name):
    with open('data/processed/'+name+'.txt') as f:
        parsing = f.read()
        parsing = parsing.split('\n')
        data = []
        for i in parsing:
            nums = i.split(",")
            holder = []
            for j in nums:
                if(len(j) < 2): continue
                a = np.array([float(elem) for m,elem in enumerate(j.split(" "))])
                holder.append(a)
            data.append(holder)
        data = data[:-1]
    return data


def plot_stuff():
    loss_epoch = {}
    fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(10,10),)
    fig.suptitle("Plots for showing paramaters with varying dimension", fontsize=16)
    row=0
    col=0
    for dim in [5]:
        epoch_loss,weights_1,weights_2 = train(dim,window_size,epochs,training_data,learning_rate)
        loss_epoch.update( {dim: epoch_loss} )

        word_similarity_scatter_plot_bigger_corpus(
            index_to_word,
            weights_1[epochs -1],
            'Vectors',
            "Plots/"
        )

        # word_similarity_scatter(
        #     index_to_word,
        #     weights_1[epochs -1],
        #     'dimension_' + str(dim) + '_epochs_' + str(epochs) + '_window_size_' +str(window_size),
        #     fig,
        #     axes[row][col]
        # )

        if col == 1:
            row += 1
            col = 0
        else:
            col += 1

    plt.savefig('Plots/Vectors' +'.png')
    plt.show()

    plot_epoch_loss('dim:',loss_epoch,'epochs_' + str(epochs) + '_window_size_' +str(window_size),'Plots/')


def print_similar_words(top_n_words,weight,msg,words_subset):

    columns=[]

    for i in range(0,top_n_words):
        columns.append('similar:' +str(i+1))

    df = pd.DataFrame(columns=columns,index=words_subset)
    df.head()

    row = 0
    overall = []
    for word in words_subset:

        #Get the similarity matrix for the word: word
        similarity_matrix = cosine_similarity(word,weight,word_to_index,vocab_size,index_to_word)
        col = 0

        #Sort the top_n_words
        words_sorted = dict(sorted(similarity_matrix.items(), key=lambda x: x[1], reverse=True)[1:top_n_words+1])

        #Create a dataframe to display the similarity matrix
        cols = [word]
        for similar_word, similarity_value in words_sorted.items():
            # df.iloc[row][col] = (similar_word, round(similarity_value, 2))
            cols.append((similar_word, round(similarity_value, 2)))
            col += 1
        row += 1
        overall.append(cols)

    print(overall)
    # styles = [dict(selector='caption',
    # props=[('text-align', 'center'),('font-size', '20px'),('color', 'red')])]
    # df = df.style.set_properties(**
    #                    {'color': 'green','border-color': 'blue','font-size':'14px'}
    #                   ).set_table_styles(styles).set_caption(msg)
    return overall

def plot_and_tables():
    loss_epoch = {}
    dataframe_sim = []

    epoch_loss,weights_1,weights_2 = train(dimension,window_size,epochs,training_data,learning_rate,'yes',50)
    loss_epoch.update( {'no': epoch_loss} )

    word_similarity_scatter_plot_bigger_corpus(
        index_to_word,
        weights_1[epochs -1],
        'Stopwords_not_removed_dimension_' + str(dimension) + '_epochs_' + str(epochs) + '_window_size_' +str(window_size),
        'Plots/'
    )

    df = print_similar_words(
        top_n_words,
        weights_1[epochs - 1],
        'sim_matrix for : Stopwords_not_removed_dimension_' + str(dimension) + '_epochs_' + str(epochs) + '_window_size_' +str(window_size),
        words_subset
    )

    dataframe_sim.append(df)
    plot_epoch_loss(
        'Stopwords_removed_',
        loss_epoch,
        'With_Stopwords_epochs_' + str(epochs) + '_window_size_' +str(window_size),
        'Plots/'
    )
    return dataframe_sim


if __name__ == "__main__":
    training_data = read_files("take1")
    index_to_word = {}
    with open("data/processed/dict.csv") as file:
     for line in file:
        (key, value) = line.split(",")
        index_to_word[int(key)] = value.replace("\n", "")

    #important constants
    epochs = 200
    top_n_words = 5
    dimension = 50
    window_size = 2
    learning_rate = 0.01

    words_subset = ['environment', 'sustainability','discussions','benefit']

    word_to_index = {v: k for k, v in index_to_word.items()}

    dataframe_sim = plot_and_tables()
    df = pd.DataFrame(dataframe_sim)
    df.to_csv("similar.csv")
