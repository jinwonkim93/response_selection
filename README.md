# Install
```bash
pip install poetry
poetry config virtualenvs.create false

poetry install
```


# Create Ubuntu corpus dataset
```bash
download original folder containing train.txt, valid.txt, text.txt
python preprocess_txt_file_to_jsonl.py --input_dir {original_folder} --output_dir ubuntu_corpus
```

# Check datasets from notebook folder
