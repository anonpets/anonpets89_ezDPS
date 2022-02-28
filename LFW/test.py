import pickle
filename = 'pipeline_4_04.sav'

# load the model from disk
clf = pickle.load(open(filename, 'rb'))

print(clf.support_vectors_.shape)
