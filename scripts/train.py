from transformers import BertForPreTraining
from transformers import AdamW
from transformers import AutoTokenizer

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from response_selection.data.dataset import UbuntuCorpusDataset


def train(args):
    
    
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, use_auth_token=True)
    bertconfig = BertConfig.from_pretrained(args.bert_model)
    model = BertForPreTraining.from_pretrained(args.bert_model, config=bertconfig)
    
    print("Loading Train Dataset", args.dataset_path)
    train_dataset = BERTDataset(Path(args.dataset_path) / "train.jsonl", tokenizer)
    
    
    # Prepare optimizer TODO
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    criterion = CrossEntropyLoss(ignore_index=-1)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=2)
    learning_rate=args.learning_rate
    before = 10
    for epoch in trange(1, int(args.num_train_epochs) + 1, desc="Epoch"):
        tr_loss = 0
        model.train()

        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration",position=0)):
            #TODO Fix loading data
            with torch.no_grad():
                batch = (item.cuda(device=device) for item in batch)
            utterances, reponse, label = batch
            optimizer.zero_grad()
            prediction_scores = model(input_ids=input_ids,attention_mask= input_mask)
            loss = criterion(prediction_scores, label)

            if step%100==0:
                print('Batch[{}] - loss: {:.6f}  batch_size:{}'.format(step, loss.item(),args.train_batch_size) )

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            else:
                loss.backward()

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                if global_step / num_train_steps < args.warmup_proportion:
                    lr_this_step = learning_rate * warmup_linear(global_step/num_train_steps, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        averloss=tr_loss/step
        print("epoch: %d\taverageloss: %f\tstep: %d "%(epoch,averloss,step))
        print("current learning_rate: ", learning_rate)
        if global_step/num_train_steps > args.warmup_proportion and averloss > before - 0.01:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.9
                learning_rate = param_group['lr']
            print("Decay learning rate to: ", learning_rate)

        before=averloss
        
        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--dataset_path",
                        default="./ubuntu_data/",
                        type=str,
                        help="The input train corpus.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default="./pretrained",
                        type=str,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--train_batch_size",
                        default=50,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=1.5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=2.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.01,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")

    args = parser.parse_args()
    train(args)
    
if __name__ == "__main__":
    main()