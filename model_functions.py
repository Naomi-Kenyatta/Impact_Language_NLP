import numpy as np
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
#nothing to run here just helper functions

vocab_size = 1024

def forward_prop(weight_inp_hidden,weight_hidden_output,target_word_vector):
    #target_word_vector = x , weight_inp_hidden =  weights for input layer to hidden layer
    hidden_layer = np.dot(weight_inp_hidden.T, target_word_vector)

    #weight_hidden_output = weights for hidden layer to output layer
    u = np.dot(weight_hidden_output.T, hidden_layer)

    y_predicted = softmax(u)

    return y_predicted, hidden_layer, u

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def backward_prop(weight_inp_hidden,weight_hidden_output,total_error, hidden_layer, target_word_vector,learning_rate):

    dl_weight_inp_hidden = np.outer(target_word_vector, np.dot(weight_hidden_output, total_error.T))
    dl_weight_hidden_output = np.outer(hidden_layer, total_error)

    # Update weights
    weight_inp_hidden = weight_inp_hidden - (learning_rate * dl_weight_inp_hidden)
    weight_hidden_output = weight_hidden_output - (learning_rate * dl_weight_hidden_output)

    return weight_inp_hidden,weight_hidden_output


def calculate_error(y_pred,context_words):

    total_error = [None] * len(y_pred)
    index_of_1_in_context_words = {}

    for index in np.where(context_words == 1)[0]:
        index_of_1_in_context_words.update ( {index : 'yes'} )

    number_of_1_in_context_vector = len(index_of_1_in_context_words)

    for i,value in enumerate(y_pred):

        if index_of_1_in_context_words.get(i) != None:
            total_error[i]= (value-1) + ( (number_of_1_in_context_vector -1) * value)
        else:
            total_error[i]= (number_of_1_in_context_vector * value)


    return  np.array(total_error)

def calculate_loss(u,ctx):

    sum_1 = 0
    for index in np.where(ctx==1)[0]:
        sum_1 = sum_1 + u[index]

    sum_1 = -sum_1
    sum_2 = len(np.where(ctx==1)[0]) * np.log(np.sum(np.exp(u)))

    total_loss = sum_1 + sum_2
    return total_loss

def train(word_embedding_dimension,window_size,epochs,training_data,learning_rate,disp = 'yes',interval=-1):
    print("training")
    weights_input_hidden = np.random.uniform(-1, 1, (vocab_size, word_embedding_dimension))
    weights_hidden_output = np.random.uniform(-1, 1, (word_embedding_dimension, vocab_size))


    #For analysis purposes
    epoch_loss = []
    weights_1 = []
    weights_2 = []

    for epoch in range(epochs):
        loss = 0

        for target,context in training_data:
            y_pred, hidden_layer, u = forward_prop(weights_input_hidden,weights_hidden_output,target)

            total_error = calculate_error(y_pred, context)

            weights_input_hidden,weights_hidden_output = backward_prop(
                weights_input_hidden,weights_hidden_output ,total_error, hidden_layer, target,learning_rate
            )

            loss_temp = calculate_loss(u,context)
            loss += loss_temp

        epoch_loss.append( loss )
        weights_1.append(weights_input_hidden)
        weights_2.append(weights_hidden_output)

        if disp == 'yes':
            if epoch ==0 or epoch % interval ==0 or epoch == epochs -1:
                print('Epoch: %s. Loss:%s' %(epoch,loss))

    return epoch_loss,np.array(weights_1),np.array(weights_2)


# Input vector, returns nearest word(s)
def cosine_similarity(word,weight,word_to_index,vocab_size,index_to_word):

    #Get the index of the word from the dictionary
    index = word_to_index[word]

    #Get the correspondin weights for the word
    word_vector_1 = weight[index]


    word_similarity = {}

    for i in range(vocab_size):

        word_vector_2 = weight[i]

        theta_sum = np.dot(word_vector_1, word_vector_2)
        theta_den = np.linalg.norm(word_vector_1) * np.linalg.norm(word_vector_2)
        theta = theta_sum / theta_den

        word = index_to_word[i]
        word_similarity[word] = theta

    return word_similarity #words_sorted
