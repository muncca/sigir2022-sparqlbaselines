from transformers import T5Tokenizer, T5ForConditionalGeneration
from flask import Flask, jsonify, request
from gevent.pywsgi import WSGIServer
from Test_T5 import Model, Test

import json
import os
import torch
import pickle
import logging
import torch.nn as nn
import argparse

logging.basicConfig(filename='sigir2022-sparql-baselines.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
torch.manual_seed(42)

argument_parser = argparse.ArgumentParser(description='SIGIR2022-SPARQL-Baselines')
argument_parser.add_argument("--port", help="port", default=9009, type=int, dest="port")
argument_parser.add_argument('--model_name',type=str,default='t5-base')
argument_parser.add_argument('--checkpoint',type=str,default=None)
argument_parser.add_argument('--device',type=int,default=0)
argument_parser.add_argument('--beam_length',type=int,default=10)
argument_parser.add_argument('--save_dir',type=str,default='base')
parsed_arguments = argument_parser.parse_args()

tester=Test(parsed_arguments)

@app.route('/api/query', methods=['POST'])
def query():
    question = request.json['question']
    raw_entities = request.json['entities']
    raw_relations = request.json['relations']

    entities = []
    for item in raw_entities:
        for uri in item["uris"]:
            entities.append(uri["uri"])

    relations = []
    for item in raw_relations:
        for uri in item["uris"]:
            relations.append(uri["uri"])

    logger.debug(question)
    logger.debug(entities)
    logger.debug(relations)

    return jsonify(tester.query(question, entities, relations))

if __name__ == '__main__':
    logger.info("Starting the HTTP server")
    http_server = WSGIServer(('', parsed_arguments.port), app)
    http_server.serve_forever()