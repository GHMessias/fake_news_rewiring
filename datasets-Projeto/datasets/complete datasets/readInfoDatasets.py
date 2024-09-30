#Code for read dataset information
import pickle
import numpy as np 

#Read dicts and load in a numpy list
documents = pickle.load(open("dict_documents.pkl", "rb"))
indexes = np.load('indexes.npy')
labels = np.load('labels.npy')

#Printing information datasets
for i in range(len(indexes)):
	index = indexes[i]
	label = labels[i]
	txt = documents[index]
	