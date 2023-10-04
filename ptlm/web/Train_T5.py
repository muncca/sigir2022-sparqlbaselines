from transformers import T5Tokenizer, T5ForConditionalGeneration
import transformers

import json

import torch
import torch.optim as optim
import pickle
import torch.nn as nn
import random

import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--split_file',type=str,default=None)
parser.add_argument('--test',type=bool,default=False)
parser.add_argument('--model_name',type=str,default='t5-base')
parser.add_argument('--checkpoint',type=str,default=None)
parser.add_argument('--device',type=int,default=0)
parser.add_argument('--save_dir',type=str,default=None)
args=parser.parse_args()


torch.manual_seed(42)

file=open(args.split_file,'rb')
data=pickle.load(file)
file.close()

final_data_test=data[0]
total_len=len(data[1])
final_data_dev,final_data=data[1][:total_len//10],data[1][total_len//10:]

class Model(nn.Module):
        def __init__(self,model_name):
                super(Model,self).__init__()
                self.model=T5ForConditionalGeneration.from_pretrained(model_name)

        def forward(self,input):
                outputs=self.model(input_ids=input['input_ids'], \
                                           labels=input['labels'], attention_mask=input['attention_mask'],output_hidden_states=True,output_attentions=True)

                return outputs.loss
                

class Train:
        def __init__(self,data,data_val,data_test,args):
                self.data=data
                self.dev_data=data_val
                self.test_data=data_test
                self.split=args.split_file.split('.')[0][-1]
                if 'mix' in args.split_file:
                    self.split='_mix'+self.split

                self.tokenizer=T5Tokenizer.from_pretrained(args.model_name)
                self.model=nn.DataParallel(Model(args.model_name),device_ids=[args.device])
                self.model.to(f'cuda:{self.model.device_ids[0]}')  
               
                self.optimizer=optim.AdamW(self.model.parameters(),lr=0.0015)
                self.lr_scheduler=transformers. \
                get_polynomial_decay_schedule_with_warmup(self.optimizer, 5000, 10000,power=0.5)

                self.iters=15000
                self.print_every=100
                self.eval_every=1000
                self.num_gpus=1
                self.eval_bs=8
                self.bs=5
                self.back_propogate=10

                self.args=args
                
                if args.test:
                    params=torch.load(args.checkpoint);
                    self.model.load_state_dict(params);
                    print('started')
                    self.test()
                else: self.train()

        def generate_batch(self):
                output=random.sample(self.data,self.bs)
                inp,label=[],[]
                for dat in output:
                        inp.append(dat[0])
                        label.append(dat[1])

                return inp,label

        def preprocess_function(self,inputs, targets):
                model_inputs=self.tokenizer(inputs, padding=True, \
                                            return_tensors='pt',max_length=512, truncation=True)
                labels=self.tokenizer(targets,padding=True,max_length=512, truncation=True)

                if True:
                    labels["input_ids"] = [
                        [(l if l != self.tokenizer.pad_token_id else -100) \
                         for l in label] for label in labels["input_ids"]
                    ]
                labels['input_ids']=torch.tensor(labels['input_ids'])
                model_inputs["labels"]=labels["input_ids"].to(f'cuda:{self.model.device_ids[0]}')
                model_inputs["input_ids"]=model_inputs["input_ids"].to(f'cuda:{self.model.device_ids[0]}')
                model_inputs["attention_mask"]=model_inputs["attention_mask"].to(f'cuda:{self.model.device_ids[0]}')

                return model_inputs

        def val(self,o):
                print('Evaluating ...')
                self.model.eval()
                acc,bs,i=0,self.eval_bs,0
                saver=[]
               
                while i<len(self.dev_data):
                    bs_=min(bs,len(self.dev_data)-i)
                    i+=bs_
                    inp,label=[],[]
                    for j in range(i-bs_,i):
                            inp.append(self.dev_data[j][0])
                            label.append(self.dev_data[j][1])

                    input=self.preprocess_function(inp,label)
                    
                    output=self.model.module.model.generate(input_ids=input['input_ids'],
                                          num_beams=8,attention_mask=input['attention_mask'], \
                                            early_stopping=True, max_length=200,output_hidden_states=True,output_attentions=True)
                    
                    out=self.tokenizer.batch_decode(output,skip_special_tokens=False)

                    for k in range(len(out)):
                            #print(out[k].replace('<pad>','').replace('</s>','').strip())
                            a1=out[k].replace('<pad>','').replace('</s>','').replace('<unk>','').replace('<s>','').strip().replace(' ','')
                            a2=label[k].strip().replace(' ','')
                            #print(a1, '       ', a2)
                            saver.append({'input':inp[k],'gold':label[k].strip(),'generated':out[k].replace('<pad>',''). \
                                          replace('</s>','').replace('<unk>','').replace('<s>','').strip()})
                            if a1==a2:
                                    acc+=1; #print('ttt')
                
                file=open(self.args.save_dir+'/split'+str(self.split)+'_dev_result'+str(o)+'.json','w')
                json.dump(saver,file)
                file.close()
                return 100*acc/len(self.dev_data)
                
        def test(self):
                self.model.eval()
                acc,bs,i=0,self.eval_bs,0
                saver=[]
               
                while i<len(self.test_data):
                    bs_=min(bs,len(self.test_data)-i)
                    i+=bs_
                    inp,label=[],[]
                    for j in range(i-bs_,i):
                            inp.append(self.test_data[j][0])
                            label.append(self.test_data[j][1])

                    input=self.preprocess_function(inp,label)
                    
                    output=self.model.module.model.generate(input_ids=input['input_ids'],
                                          num_beams=10,attention_mask=input['attention_mask'], \
                                            early_stopping=True, max_length=100,output_hidden_states=True,output_attentions=True,num_return_sequences=10)

                    out=self.tokenizer.batch_decode(output,skip_special_tokens=False)

                    for k in range(len(out)):
                            #print(out[k].replace('<pad>','').replace('</s>','').strip())
                            a1=out[k].replace('<pad>','').replace('</s>','').strip().replace(' ','')
                            a2=label[k].strip().replace(' ','')
                            #print(a1, '       ', a2)
                            saver.append({'input':inp[k],'gold':label[k].strip(),'generated':out[k].replace('<pad>',''). \
                                          replace('</s>','').strip()})
                            if a1==a2:
                                    acc+=1; #print('ttt')
                
                file=open('split'+str(self.split)+'_test_result.json','w')
                json.dump(saver,file)
                file.close()
                print(100*acc/len(self.test_data))


        def train(self):

                scalar=0
                for i in range(self.iters):
                        self.model.train()
                        inp,label=self.generate_batch()
                        input=self.preprocess_function(inp,label)
                        loss=self.model(input)

                        scalar+=loss.mean().item()
                        if(i+1)%self.print_every==0:
                                print('iteration={}, training loss={}'.format(i+1,scalar/self.print_every))
                                scalar=0
                        if(i+1)%self.eval_every==0:
                                acc=self.val(i+1)
                                print('validation acc={}'.format(acc))

                                torch.save(self.model.state_dict(),self.args.save_dir+'/split'+str(self.split)+'_checkpoint'+str(i+1)+'.pth')
                        
                        loss/=self.back_propogate
                        loss.mean().backward()
                        if (i+1)%self.back_propogate:
                                self.optimizer.step();
                                self.lr_scheduler.step();
                                self.optimizer.zero_grad()

trainer=Train(final_data,final_data_dev,final_data_test,args)
