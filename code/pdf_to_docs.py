import os
import sys
import argparse
import pickle
from llama_index.readers.docling import DoclingReader
from llama_index.core import SimpleDirectoryReader



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, required=True, help="Path to directory with PDF files to import")
    parser.add_argument('--output_file', type=str, required=True, help="Where to save the imported and serialized documents - should be in .spacy format")
    args = parser.parse_args()

    directory = args.directory
    out_file = args.output_file
    
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

    if not out_file.lower().endswith('.pkl'):
        raise ValueError(f"Output file '{out_file}' should be a .pkl file.")

    # Read and convert PDF to md (default)
    reader = DoclingReader()
    dir_reader = SimpleDirectoryReader(
        input_dir=directory,
        file_extractor={".pdf": reader},
    )
    docs = dir_reader.load_data()

    # Save parsed files
    with open(out_file, 'wb') as handle:
        pickle.dump(docs, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()