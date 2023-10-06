from transformers import T5Tokenizer, T5ForConditionalGeneration
from flask import Flask, jsonify
from gevent.pywsgi import WSGIServer

import json
import os
import torch
import pickle
import logging
import torch.nn as nn


import argparse

class Model(nn.Module):
        def __init__(self,model_name):
                super(Model,self).__init__()
                self.model=T5ForConditionalGeneration.from_pretrained(model_name)

        def forward(self,input):
                outputs=self.model(input_ids=input['input_ids'], \
                                           labels=input['labels'],  \
                                           attention_mask=input['attention_mask'], \
                                           output_hidden_states=True,output_attentions=True)

                return outputs.loss
                

class Test:
        def __init__(self,data_test,args):
                self.test_data=data_test
                self.split=args.split_file.split('.')[0][-1]
                if 'mix' in args.split_file:
                    self.split='_mix'+self.split

                self.tokenizer=T5Tokenizer.from_pretrained(args.model_name)
                self.model=nn.DataParallel(Model(args.model_name),device_ids=[args.device])
                if torch.cuda.is_available():
                    self.model.to(f'cuda:{self.model.device_ids[0]}')  
                
                self.num_gpus=1
                self.eval_bs=8
                self.beam=args.beam_length
                
                self.args=args

                if torch.cuda.is_available():
                    params=torch.load(args.save_dir+'/'+args.checkpoint)                    
                else:
                    params=torch.load(args.save_dir+'/'+args.checkpoint, map_location=torch.device('cpu'))

                self.model.load_state_dict(params);
                print('started')
                
                 
                self.sparql_vocab=['?x','{','}','?uri','SELECT', 'DISTINCT', 'COUNT', '(', ')',  \
              'WHERE', '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>', \
              '.','ASK','[DEF]','<http://dbpedia.org/ontology/','<http://dbpedia.org/property/', \
              '<http://dbpedia.org/resource/']
                self.vocab_dict={}
                for i in range(len(self.sparql_vocab)):
                    self.vocab_dict['<extra_id_'+str(i)+'>']=self.sparql_vocab[i]
                self.vocab_dict['<extra_id_17>']=''
                
                #self.test()
                
        def readable(self,string):
            for key in self.vocab_dict:
                string=string.replace(key,' '+self.vocab_dict[key]+' ')
            string=string.replace('  ',' ')
            vals=string.split()
                
            for i,val in enumerate(vals):
                if val=='<http://dbpedia.org/ontology/' or val=='<http://dbpedia.org/property/'  \
                or val=='<http://dbpedia.org/resource/':
                    if i<len(vals)-1:
                        vals[i]=val+vals[i+1]+'>'
                        vals[i+1]=''
                        
            return ' '.join(vals).strip().replace('  ',' ')

        def preprocess_function(self,inputs):
                model_inputs=self.tokenizer(inputs, padding=True, \
                                            return_tensors='pt',max_length=512, truncation=True)
                if torch.cuda.is_available():
                    #model_inputs["labels"]=labels["input_ids"].to(f'cuda:{self.model.device_ids[0]}')
                    model_inputs["input_ids"]=model_inputs["input_ids"].to(f'cuda:{self.model.device_ids[0]}')
                    model_inputs["attention_mask"]=model_inputs["attention_mask"].to(f'cuda:{self.model.device_ids[0]}')
                else:
                    #model_inputs["labels"]=labels["input_ids"]
                    model_inputs["input_ids"]=model_inputs["input_ids"]
                    model_inputs["attention_mask"]=model_inputs["attention_mask"]

                return model_inputs
        
        def prepare_input(self, question, entities, relations):
            input = question+" "

            for entity_url in entities:
                entity = entity_url.rsplit('/', 1)[-1]
                input += "<extra_id_13> <extra_id_16> "+entity+" "

            for relation_url in relations:
                relation = relation_url.rsplit('/', 1)[-1]
                if '/ontology/' in relation_url:
                    input += "<extra_id_13> <extra_id_14> "+relation+" "
                elif '/property/' in relation_url:
                    input += "<extra_id_13> <extra_id_15> "+relation+" "
                else:
                    print("wtf is this relation???")

            print(input)            
            return input

        def test(self, question, entities, relations):
                self.model.eval()                
                inp=[]                

                #question = "What is the age of Barack Obama?"
                #entities = ["http://dbpedia.org/resource/Barack_Obama"]
                #relations = ["http://dbpedia.org/ontology/birthDate"]

                inp.append(self.prepare_input(question, entities, relations))

                input=self.preprocess_function(inp)

                output=self.model.module.model.generate(input_ids=input['input_ids'],
                                      num_beams=self.beam,attention_mask=input['attention_mask'], \
                                        early_stopping=True, max_length=100,num_return_sequences=self.beam)
                
                out=self.tokenizer.batch_decode(output,skip_special_tokens=False)


                result = {}
                result['queries']=[]
                for k in range(len(out)//self.beam):               
                    for s in range(self.beam):
                        result['queries']. \
                        append(self.readable(out[int(k*self.beam+s)].replace('<pad>','').replace('</s>','').strip()))
                
                print(result)
                return result