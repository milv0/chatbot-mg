# -*- coding: utf-8 -*-
import argparse
import logging

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel


parser = argparse.ArgumentParser(description='Simsimi based on KoGPT-2')

parser.add_argument('--chat',
                    action='store_true',
                    default=False,
                    help='response generation on given user input')

parser.add_argument('--sentiment',
                    type=str,
                    default='0',
                    help='sentiment for system. 0 is neutral, 1 is negative, 2 is positive.')

parser.add_argument('--model_params',
                    type=str,
                    default='model_chp_v2/model_-last.ckpt',
                    help='model binary for starting chat')

parser.add_argument('--train',
                    action='store_true',
                    default=False,
                    help='for training')

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)

U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 


class CharDataset(Dataset):
    def __init__(self, chats, max_len=32):
        self._data = chats
        self.first = True
        self.q_token = U_TKN
        self.a_token = S_TKN
        self.sent_token = SENT
        self.bos = BOS
        self.eos = EOS
        self.mask = MASK
        self.pad = PAD
        self.max_len = max_len
        # self.tokenizer = TOKENIZER 
        self.tokenizer = TOKENIZER if TOKENIZER is not None else PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK)

    def __len__(self):
        # print(len(self._data))
        return len(self._data)

    def __getitem__(self, idx):
        print("Debugging: __getitem__ method is called.")

        turn = self._data.iloc[idx]
        q = turn['Q']
        a = turn['A']
        sentiment = str(turn['label'])
        q_toked = self.tokenizer.tokenize(self.q_token + q + \
                                        self.sent_token + sentiment)   
        q_len = len(q_toked)
        a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos)
        a_len = len(a_toked)

        # 디버깅을 위한 출력 추가
        print(f"q_toked: {q_toked}")
        print(f"a_toked: {a_toked}")

        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len
            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len/2)):]
                q_len = len(q_toked)
                a_len = self.max_len - q_len
                assert a_len > 0
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)
            assert a_len == len(a_toked), f'{a_len} ==? {len(a_toked)}'
        
        # [mask, mask, ...., mask, ..., <bos>,..A.. <eos>, <pad>....]
        labels = [self.mask,] * q_len + a_toked[1:]

        # 디버깅을 위한 출력 추가
        print(f"labels: {labels}")

        if self.first:
            print("contexts : {}".format(q))
            print("toked ctx: {}".format(q_toked))
            print("response : {}".format(a))
            print("toked response : {}".format(a_toked))
            print('labels {}'.format(labels))
            self.first = False
        
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        # self.max_len
        # 디버깅을 위한 출력 추가
        print(f"mask: {mask}")

        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]
        
        # 디버깅을 위한 출력 추가
        print(f"labels_ids: {labels_ids}")

        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]

        # 디버깅을 위한 출력 추가
        print(f"token_ids: {token_ids}")

        return (token_ids, np.array(mask), labels_ids)


class KoGPT2Chat(LightningModule):
    def __init__(self, hparams, **kwargs):
        super(KoGPT2Chat, self).__init__()
        self.hparams = hparams
        self.neg = -1e18
        self.kogpt2 = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max-len',
                            type=int,
                            default=64,
                            help='max sentence length on input (default: 32)')

        parser.add_argument('--batch-size',
                            type=int,
                            default=96,
                            help='batch size for training (default: 96)')
        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')
        return parser

    def forward(self, inputs):
        # (batch, seq_len, hiddens)
        output = self.kogpt2(inputs, return_dict=True)
        return output.logits

    def training_step(self, batch, batch_idx):
        token_ids, mask, label = batch
        out = self(token_ids)
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        loss = self.loss_function(mask_out.transpose(2, 1), label)
        loss_avg = loss.sum() / mask.sum()
        self.log('train_loss', loss_avg)
        return loss_avg

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.lr)

        # warm up lr
        train_dataloader_length = len(self.train_dataloader())
        # print(f"Train DataLoader Length: {train_dataloader_length}")

        num_train_steps = train_dataloader_length * self.hparams.max_epochs  # 수정된 부분
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def _collate_fn(self, batch):
        data = torch.LongTensor([item[0] for item in batch])
        mask = torch.LongTensor([item[1] for item in batch])
        label = torch.LongTensor([item[2] for item in batch])
        return data, mask, label

    def train_dataloader(self):
        try:
            # Load data
            data = pd.read_csv('chatbot_dataset-v2.csv')

            print("Loaded data:\n", data.head())

            if data.empty:
                raise ValueError("Loaded dataset is empty. Check the data file.")
            
            # Initialize dataset
            self.train_set = CharDataset(data, max_len=self.hparams.max_len)
            if not self.train_set:  # 수정된 부분
                raise ValueError("Training dataset is empty after CharDataset initialization.")
            else:
                print('Train_set length: ', len(self.train_set))
            
            # Create DataLoader
            train_dataloader = DataLoader(
                self.train_set, batch_size=self.hparams.batch_size, num_workers=0, # 2-> 0 수정
                shuffle=True, collate_fn=self._collate_fn)
            
            print("Train DataLoader:", train_dataloader)
            print("Train DataLoader length:", len(train_dataloader))

            return train_dataloader
        except Exception as e:
            print(f"Error in train_dataloader: {e}")
            raise

    def chat(self, sent='0', user_input=None):
        tok = TOKENIZER
        sent_tokens = tok.tokenize(sent)
        with torch.no_grad():
            q = user_input.strip() if user_input else input('user > ')
            a = ''
            while 1:
                input_ids = torch.LongTensor(tok.encode(U_TKN + q + SENT + sent + S_TKN + a)).unsqueeze(dim=0)
                pred = self(input_ids)
                gen = tok.convert_ids_to_tokens(
                    torch.argmax(
                        pred,
                        dim=-1).squeeze().numpy().tolist())[-1]
                if gen == EOS:
                    break
                a += gen.replace('▁', ' ')
            
            print("Chatbot > {}".format(a.strip()))

            return q

parser = KoGPT2Chat.add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()
logging.info(args)

if __name__ == "__main__":
    if args.train:
        checkpoint_callback = ModelCheckpoint(
            dirpath='model_chp_v2',
            filename='{epoch:02d}-{train_loss:.2f}',
            verbose=True,
            save_last=True,
            monitor='train_loss',
            mode='min',
            prefix='model_'
        )
        ### CUDA 메모리 할당기 설정 변경
        torch.cuda.memory._set_allocator_settings('expandable_segments:False')

        model = KoGPT2Chat(args)
        # model.train()
        trainer = Trainer.from_argparse_args(
            args,
            checkpoint_callback=checkpoint_callback, gradient_clip_val=1.0,gpus=4,
             # attention_mask를 사용하도록 추가
            accumulate_grad_batches=8,  # 그라디언트 누적 배치
            precision=16,  # Mixed Precision 사용
            log_gpu_memory='all',  # GPU 메모리 로깅
            )
        
        # 모델을 GPU로 옮기고 병렬처리를 위해 지정된 GPU에 모델을 배치
        model.cuda()

        trainer.fit(model)

        best_model_path = checkpoint_callback.best_model_path
        # logging.info('best model path {}'.format(best_model_path))
        print('Best model path: {}'.format(best_model_path))    


    if args.chat:
        model = KoGPT2Chat.load_from_checkpoint(args.model_params)
        model.eval()  # 모델을 evaluation 모드로 설정
        while True:
            user_input = input('user > ')
            if user_input.lower() in ['exit', 'quit', 'q']:
                # 사용된 모델 이름
                print('Current model path: {}'.format(args.model_params))
                break  # 종료 조건
            model.chat(sent=args.sentiment, user_input=user_input)



