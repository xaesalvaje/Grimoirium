import os
import sys
from scripts.train_model import train_word2vec_model
from scripts.combine_pdf import merge_pdfs
from scripts.convert_to_text import convert_to_txt
from scripts.preprocess_text import preprocess_text
from scripts.create_master_corpus import create_master_corpus
from scripts.combine_pdf import merge_pdfs

# merge PDF files into one
merged_pdf = merge_pdfs('./data/input', './data/output/merged.pdf')
print("PDF files merged successfully!")

# preprocess merged PDF into a text file
# this script takes the merged PDF and preprocesses it
# for training the Word2Vec model
train_word2vec_model('./data/output/merged.pdf', './models/')

sys.path.append('./scripts/')

# set up file paths
input_dir = "corpus/input/"
output_dir = "corpus/output/"
processed_dir = "corpus/processed/"
raw_dir = "corpus/raw/"

# create directories if not exist
if not os.path.exists(input_dir):
    os.makedirs(input_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

if not os.path.exists(raw_dir):
    os.makedirs(raw_dir)


def run_grimoirium():
    # merge pdfs in input directory
    merge_pdfs(input_dir, output_dir)

    # convert pdfs to txt
    convert_to_txt(output_dir, processed_dir)

    # preprocess text files
    preprocess_text(processed_dir, raw_dir)
    
    # train Word2Vec model
    train_word2vec_model(processed_dir, model_dir, logs_dir)

    # create master corpus
    create_master_corpus(raw_dir)


if __name__ == "__main__":
    run_grimoirium()
