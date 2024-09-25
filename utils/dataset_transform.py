import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import argparse
import os

# parser = argparse.ArgumentParser()

# parser.add_argument('--compute_mean', action = 'store_true')
# parser.add_argument('--process_data', action = 'store_true')
# parser.add_argument('--input_path')
# parser.add_argument('--output_path')

# args = parser.parse_args()

def process_data(input_path, output_path, stopwords_language = 'portuguese'):
    def preprocess_text(text):

        # # uncoment if you want do download the stopwords
        # nltk.download('stopwords')
        
        # Converter para lowercase
        text = text.lower()

        # Remover links
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)

        # Remover números
        text = re.sub(r'\d+', '', text)

        # Remover pontuações (opcional, ajuste conforme necessário)
        text = re.sub(r'[^\w\s]', '', text)

        # Tokenizar o texto (separar por palavras)
        words = text.split()

        # Remover stopwords em inglês
        stop_words = set(stopwords.words(stopwords_language))
        words = [word for word in words if word not in stop_words]

        # Aplicar PorterStemmer
        ps = PorterStemmer()
        stemmed_words = [ps.stem(word) for word in words]

        return ' '.join(stemmed_words)

    def process_files_in_directory(input_directory, output_directory):
        # Cria a pasta de saída se ela não existir
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        for filename in os.listdir(input_directory):
            if filename.endswith('.txt'):
                input_file_path = os.path.join(input_directory, filename)
                with open(input_file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                    # Pré-processamento do texto
                    processed_content = preprocess_text(content)

                    # Caminho para salvar o arquivo processado
                    output_file_path = os.path.join(output_directory, filename)

                    # Salvar o conteúdo processado no novo arquivo
                    with open(output_file_path, 'w', encoding='utf-8') as output_file:
                        output_file.write(processed_content)

                    print(f"Arquivo processado salvo em: {output_file_path}")

    # Exemplo de uso
    input_directory = input_path  # Pasta com os arquivos originais
    output_directory = output_path  # Pasta onde os arquivos processados serão salvos

    process_files_in_directory(input_directory, output_directory)

    print('files processed')

    return

