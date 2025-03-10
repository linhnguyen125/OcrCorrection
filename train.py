import multiprocessing
import os
import time
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.dataset import BasicDataset, ClusterRandomSampler, Collator
from model.seq2seq import Seq2Seq
from model.vocab import Vocab
from optim.loss import LabelSmoothingLoss
from tool.logger import Logger
from config import alphabet
from tool.translate import translate
from tool.utils import batch_to_device, compute_accuracy


class Trainer(object):
    def __init__(self):
        self.batch_size = 768
        self.num_iters = 50000
        self.valid_every = 2000
        self.print_every = 100
        self.lr = 0.0001
        self.logger = Logger('./train.log')
        
        # create vocab model
        self.vocab = Vocab(alphabet)
        
        # create model
        weight_path = './weights/seq2seq_0.pth'
        self.model = Seq2Seq(len(alphabet), encoder_hidden=256, decoder_hidden=256)
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = LabelSmoothingLoss(len(alphabet), 0).cuda(1)
        self.model = self.model.to(device=self.device)
        self.weight_path = weight_path
        
        # load pretrain weights
        if os.path.exists(weight_path):
            print('Load weight from: ', weight_path)
            self.load_weights(weight_path)
            
        # create optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-09)
        self.scheduler = OneCycleLR(self.optimizer, max_lr=self.lr, total_steps=self.num_iters, pct_start=0.1)
        
        # create dataset
        self.train_dataset = BasicDataset('./lmdb/train_lmdb')
        self.val_dataset = BasicDataset('./lmdb/val_lmdb')
        
        print('The number of train data: ', len(self.train_dataset))
        print('The number of val data: ', len(self.val_dataset))
        
        # create dataloader
        self.train_loader = self.create_dataloader(self.train_dataset, True)
        self.val_loader = self.create_dataloader(self.val_dataset, False)
        self.sample = 50000
            
    def load_weights(self, filename):
        state_dict = torch.load(filename, map_location=torch.device(self.device))

        for name, param in self.model.named_parameters():
            if name not in state_dict:
                print('{} not found'.format(name))
            elif state_dict[name].shape != param.shape:
                print(
                    '{} missmatching shape, required {} but found {}'.format(name, param.shape, state_dict[name].shape))
                del state_dict[name]

        self.model.load_state_dict(state_dict, strict=False)
        
    def save_weights(self, filename, fold_id):
        path, _ = os.path.split(filename)
        os.makedirs(path, exist_ok=True)
        if self.sample != len(self.val_loader):
            torch.save(self.model.state_dict(), './weights/seq2seq' + '_' + str(fold_id) + '.pth')
        else:
            torch.save(self.model.state_dict(), filename)
            
    def create_dataloader(self, dataset, shuffle):
        # create train and val dataloader
        sampler = ClusterRandomSampler(dataset, self.batch_size, shuffle)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, collate_fn=Collator(), shuffle=False, num_workers=0, pin_memory=True)

        return data_loader
    
    def train(self):
        total_loss = 0
        best_acc = 0
        global_step = 0
        total_time = 0
        data_iter = iter(self.train_loader)
        best_fold_acc = [0] * (len(self.val_loader) // (self.sample // self.batch_size) + 1)

        for _ in tqdm(range(self.num_iters), desc='Training...', ncols=100):
            self.model.train()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            global_step += 1
            start = time.time()
            loss = self.train_step(batch)
            total_time += time.time() - start
            total_loss += loss

            if global_step % self.print_every == 0:
                info = 'step: {:06d}, train_loss: {:.4f}, gpu_time: {}'.format(global_step, total_loss / self.print_every, total_time)
                print(info)
                self.logger.log(info)
                total_loss = 0
                total_time = 0

            if global_step % self.valid_every == 0:
                selected_fold = 0

                # validate
                val_loss = self.validate(selected_fold)
                acc_full_seq, acc_per_char = self.precision(selected_fold)

                if self.sample != len(self.val_dataset):
                    if acc_full_seq > best_fold_acc[selected_fold]:
                        self.save_weights(self.weight_path, selected_fold)
                        best_fold_acc[selected_fold] = acc_full_seq
                else:
                    if acc_full_seq > best_acc:
                        self.save_weights(self.weight_path, selected_fold)
                        best_acc = acc_full_seq

                print("==============================================================================")
                info = "val_loss: {:.4f}, full_seq_acc: {:.4f}, word_acc: {:.4f}".format(val_loss, acc_full_seq, acc_per_char)
                print(info)
                self.logger.log(info)
                print("==============================================================================")
                
    def validate(self, fold_id):
        self.model.eval()
        total_loss = []

        if self.sample != len(self.val_dataset):
            start = fold_id * (self.sample // self.batch_size)
            end = (fold_id + 1) * (self.sample // self.batch_size)

            if end > len(self.val_loader):
                end = len(self.val_loader)

        valdata_iter = iter(self.val_loader)
        with torch.no_grad():
            pbar = tqdm(range(len(self.val_loader)), ncols = 100, desc='Computing loss on {}th fold data..'.format(fold_id))
            for step in pbar:
                try:
                    batch = next(valdata_iter)
                except StopIteration:
                    valdata_iter = iter(self.val_loader)
                    batch = next(valdata_iter)

                if self.sample != len(self.val_dataset):
                    if step < start:
                        continue
                    elif step >= end:
                        break

                texts, tgt_input, tgt_output, tgt_padding_mask = batch
                texts, tgt_input, tgt_output, tgt_padding_mask = batch_to_device(texts, tgt_input, tgt_output, tgt_padding_mask, self.device)
                outputs = self.model(texts, tgt_input)

                outputs = outputs.flatten(0, 1)
                tgt_output = tgt_output.flatten()
                loss = self.criterion(outputs, tgt_output)
                total_loss.append(loss.item())
                del outputs
                del loss

        val_loss = np.mean(total_loss)
        self.model.train()

        return val_loss
    
    def precision(self, fold_id):
        pred_sents = []
        actual_sents = []

        if self.sample != len(self.val_dataset):
            start = fold_id * (self.sample // self.batch_size)
            end = (fold_id + 1) * (self.sample // self.batch_size)

            if end > len(self.val_loader):
                end = len(self.val_loader)

        valdata_iter = iter(self.val_loader)
        with torch.no_grad():
            pbar = tqdm(range(len(self.val_loader)), ncols = 100, desc='Computing accuracy on {}th fold data..'.format(fold_id))
            for step in pbar:
                try:
                    batch = next(valdata_iter)
                except StopIteration:
                    valdata_iter = iter(self.val_loader)
                    batch = next(valdata_iter)

                if self.sample != len(self.val_dataset):
                    if step < start:
                        continue
                    elif step >= end:
                        break

                texts, tgt_input, tgt_output, tgt_padding_mask = batch
                texts, tgt_input, tgt_output, tgt_padding_mask = batch_to_device(texts, tgt_input, tgt_output, tgt_padding_mask, self.device)
                actual_sent = self.vocab.batch_decode(tgt_output.tolist())
                translated_sentence = translate(texts, self.model, self.device)
                pred_sent = self.vocab.batch_decode(translated_sentence.tolist())
                pred_sents.extend(pred_sent)
                actual_sents.extend(actual_sent)

        acc_full_seq = compute_accuracy(actual_sents, pred_sents, mode='full_sequence')
        acc_per_char = compute_accuracy(actual_sents, pred_sents, mode='word')

        return acc_full_seq, acc_per_char
    
    def train_step(self, batch):
        self.model.train()

        # get the inputs
        texts, tgt_input, tgt_output, tgt_padding_mask = batch
        texts, tgt_input, tgt_output, tgt_padding_mask = batch_to_device(texts, tgt_input, tgt_output, tgt_padding_mask, self.device)

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize + scheduler
        outputs = self.model(texts, tgt_input)

        outputs = outputs.flatten(0, 1)
        tgt_output = tgt_output.flatten()
        loss = self.criterion(outputs, tgt_output)
        try:
            loss.backward()
        except RuntimeError as e:
            print(f"Error during backward pass: {e}")
            print(f"Loss: {loss}")
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    print(f"Parameter {name} has no gradient")
                else:
                    print(f"Parameter {name} gradient: {param.grad}")
            raise e
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
        self.scheduler.step()
        loss_item = loss.item()
        
        # Giải phóng bộ nhớ GPU không sử dụng
        torch.cuda.empty_cache()

        return loss_item
    
if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Khởi tạo và chạy Trainer
    Trainer().train()