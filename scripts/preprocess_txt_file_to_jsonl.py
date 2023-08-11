import json
import argparse
from pathlib import Path


def parse_textfile_to_jsonl(file_path: Path, output_path: Path):

    response_datasets = []
    with open(file_path, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break
            line_list = line.split("\t")
            label, utterances, response = line_list[0], line_list[1:-1], line_list[-1]
            response_datasets.append({"label":label, "utterances":utterances, "response":response})

    with open(output_path, "w") as json_file:
        for metatdata in response_datasets:
            json.dump(metatdata, json_file)
            json_file.write("\n")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess raw text file to jsonl')
    parser.add_argument('--input_dir', required=True, help='input folder path containing train.txt, valid.txt, test.txt')
    parser.add_argument('--output_dir', required=True, help='output folder path')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("Preprocess train dataset...")
    input_train_file = input_dir / "train.txt"
    output_train_file = output_dir / "train.jsonl"
    parse_textfile_to_jsonl(input_train_file, output_train_file)
    print("Done..!!!")
    
    print("Preprocess valid dataset...")
    input_valid_file = input_dir / "valid.txt"
    output_valid_file = output_dir / "valid.jsonl"
    parse_textfile_to_jsonl(input_valid_file, output_valid_file)
    print("Done..!!!")
    
    print("Preprocess test dataset...")
    input_test_file = input_dir / "test.txt"
    output_test_file = output_dir / "test.jsonl"
    parse_textfile_to_jsonl(input_test_file, output_test_file)
    print("Done..!!!")