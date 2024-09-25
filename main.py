'''
Main file of the experiments
'''

from runners.doc2vec import doc2vec_matrix

def main():
    matrix = doc2vec_matrix('datasets/FakeBr/fake')

if __name__ == '__main__':
    main()

