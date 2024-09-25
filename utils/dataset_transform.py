import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--compute_mean', action = 'store_true')
parser.add_argument('--process_data', action = 'store_true')
parser.add_argument('--input_path')
parser.add_argument('--output_path')

args = parser.parse_args()

if args.process_data:
    # Certifique-se de baixar as stopwords antes de rodar o código
    # nltk.download('stopwords')

    def preprocess_text(text):
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
        stop_words = set(stopwords.words('portuguese'))
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
    input_directory = args.input_path  # Pasta com os arquivos originais
    output_directory = args.output_path  # Pasta onde os arquivos processados serão salvos

    process_files_in_directory(input_directory, output_directory)

if args.compute_mean:

    def count_words_in_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            words = content.split()
            return len(words)

    def calculate_average_words(directory_path):
        total_words = 0
        file_count = 0

        # Itera por todos os arquivos da pasta
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory_path, filename)
                word_count = count_words_in_file(file_path)
                total_words += word_count
                file_count += 1

        if file_count == 0:
            return 0  # Evita divisão por zero

        return total_words / file_count

    # Exemplo de uso
    directory = args.input_path
    average = calculate_average_words(directory)
    print(f'A média de palavras por arquivo é: {average:.2f}')

