import torch
from torch.utils.data import Dataset
import jsonlines


SPECIAL_TOKENS = ["[CLS]", "[SEP]"]

def load_jsonl_to_list_dict(jsonl_path):
    list_dict = []
    with jsonlines.open(jsonl_path) as f:
        for line in f:
            list_dict.append(line)
    return list_dict

class UbuntuCorpusDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.data = load_jsonl_to_list_dict(data_path)
        self.tokenizer = tokenizer
        self.pad = tokenizer.pad_token_id
    
    def __len__(self):
        return len(self.data)
    
    def process(self, data):
        history, response, label = data['utterances'], data['response'], data['label']
        history_tokenized = [self.tokenizer(utterance, padding="max_length", truncation=True) for utterance in history]
        response_tokenized = self.tokenizer(response, padding="max_length", truncation=True)
        return {"utterances": history_tokenized, "response":response_tokenized, "label":label}
        
    
    def __getitem__(self, index):
        return self.process(self.data[index])