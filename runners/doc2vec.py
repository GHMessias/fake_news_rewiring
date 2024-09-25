import os
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

def doc2vec_matrix(path: str, train = True, save_model = False, save_model_path = None, model_path = None, ):
    # Caminho da pasta onde estão os arquivos de texto

    # Lista de documentos processados
    documents = []

    # Lê os arquivos .txt da pasta
    for filename in sorted(os.listdir(path)):
        if filename.endswith(".txt"):
            filepath = os.path.join(path, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                # Tokeniza e cria o TaggedDocument para cada arquivo
                documents.append(TaggedDocument(words=word_tokenize(content.lower()), tags=[filename]))

    if not train:
        # Carregando o modelo treinado (se necessário)
        model = Doc2Vec.load(model_path)
    else:
    # Treinando o modelo Doc2Vec
        model = Doc2Vec(documents, vector_size=500, window=8, alpha = 0.025, min_alpha = 0.0001, epochs = 100, dm_mean = 1, dm = 1)

    if save_model:
        model.save(save_model_path)

    # Criando a matriz de vetores de cada documento
    matriz_vetores = [model.dv[doc.tags[0]] for doc in documents]

    return matriz_vetores
