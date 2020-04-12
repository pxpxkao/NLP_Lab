import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import subprocess
from models import *
from utils import *
from attention import *


class Solver():
    def __init__(self, args):
        self.args = args

        self.data_utils = data_utils(args)
        self.model = self.make_model(self.data_utils.vocab_size, 4)

        self.model_dir = make_save_dir(args.model_dir)


    def make_model(self, src_vocab, N=6, 
            d_model=512, d_ff=2048, h=8, dropout=0.1):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        generator = Generator(d_model, 8)
        word_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
        model = NEREncoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            word_embed,
            generator)
        
        # This was important from their code. 
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
        return model.cuda()


    def train(self):
        data_yielder = self.data_utils.data_yielder(self.args.train_file, self.args.train_tgt_file, num_epoch = 100)

        optim = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
        total_loss = []
        start = time.time()
        
        for step in range(1000000):
            self.model.train()
            
            batch = data_yielder.__next__()
            
            out = self.model.forward(batch['src'].long(), 
                            batch['src_mask'])
            pred = out.topk(1, dim=-1)[1].squeeze().detach().cpu().numpy()
            gg = batch['src'].long().detach().cpu().numpy()
            yy = batch['y'].long().detach().cpu().numpy()
            loss = self.model.loss_compute(out, batch['y'].long())
            loss.backward()
            optim.step()
            optim.zero_grad()
            total_loss.append(loss.detach().cpu().numpy())
            
            if step % 500 == 1:
                elapsed = time.time() - start
                print("Epoch Step: %d Loss: %f Time: %f" %
                        (step, np.mean(total_loss), elapsed))
                print('src:',self.data_utils.id2sent(gg[0]))
                print('tgt:',self.data_utils.id2label(yy[0]))
                print('pred:',self.data_utils.id2label(pred[0]))

                start = time.time()
                total_loss = []
                print()

            if step % 1000 == 1:
                self.model.eval()
                val_yielder = self.data_utils.data_yielder(self.args.valid_file, self.args.valid_tgt_file)
                total_loss = []
                for batch in val_yielder:
                    batch['src'] = batch['src'].long()
                    batch['src_extended'] = batch['src_extended'].long()
                    # print(len(batch['oov_list']))
                    out = self.model.forward(batch['src'].long(),
                            batch['src_mask'])
                    loss = self.model.loss_compute(out, batch['y'].long())
                    total_loss.append(loss.item())
                print('=============================================')
                print('Validation Result -> Loss : %6.6f' %(sum(total_loss)/len(total_loss)))
                print('=============================================')
                
                w_step = int(step/10000)
                if self.args.load_model:
                    w_step += (int(self.args.load_model.split('/')[-1][0]))
                print('Saving ' + str(w_step) + 'k_model.pth!\n')
                model_name = str(w_step) + 'k_' + '%6.6f'%(sum(total_loss)/len(total_loss)) + 'model.pth'
                state = {'step': step, 'state_dict': self.model.state_dict()}

                torch.save(state, os.path.join(self.model_dir, model_name))

    def _test(self):
        #prepare model
        path = self.args.load_model
        max_len = 50
        state_dict = torch.load(path)['state_dict']
        model = self.model
        model.load_state_dict(state_dict)
        pred_dir = make_save_dir(self.args.pred_dir)
        filename = self.args.filename

        #start decoding
        data_yielder = self.data_utils.data_yielder(self.args.test_file, self.args.test_tgt_file)
        total_loss = []
        start = time.time()

        #file
        f = open(os.path.join(pred_dir, filename), 'w')

        self.model.eval()

        for batch in data_yielder:
            #print(batch['src'].data.size())
            out = self.model.forward(batch['src'].long(), batch['src_mask'])
            
            out = torch.argmax(out, dim = 2)
            #print(out.size())

            for i in range(out.size(0)):
                nonz = torch.nonzero(batch['src_mask'][i])
                print(nonz)
                idx = nonz[-1][1].item()
                sentence = self.data_utils.id2label(out[i][:idx], True)
                #print(l[1:])
                f.write(sentence)
                f.write("\n")