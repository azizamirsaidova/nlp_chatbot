from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from datasets import load_metric
from pprint import pprint

import torch
import math
import time
import sys
import json
import numpy as np
import pandas as pd

def make_woz_datasets(bKnowledge):
    
    if bKnowledge:
        out_names = ['woz.train_c.txt','woz.valid_c.txt','woz.test_c.txt']
    else:
        out_names = ['woz.train_b.txt','woz.valid_a.txt','woz.test_a.txt','woz.valid_b.txt','woz.test_b.txt']
    max_ins = [18,2,2,2,2]
    
    count = 0
    counts = []
    for dataset in range(len(out_names)):
        fout = open(out_names[dataset],'wt')
        for dialog in range(1,max_ins[dataset],1):
            file_name = '/content/gdrive/MyDrive/nlp/multiwoz/train/dialogues_%03d.json' % dialog
            print(file_name)
            with open(file_name) as f:
                data = json.load(f)
            for dialogue in data:
                if len(dialogue['services']) == 1:
                    if dialogue['services'][0] == 'restaurant':
                        prev_speaker = ''
                        prev_utterance = ''
                        for turn in dialogue['turns']:
                            count = count + 1
                            speaker = turn['speaker']
                            utterance = turn['utterance']
                            
                            for frame in turn['frames']:
                                if frame['service'] == 'restaurant':
                                    knowledge = ''
                                    try:
                                        knowledge = '[KNOWLEDGE] '
                                        for slot in frame['slots']:
                                            temp = '%s [EQUAL] %s [SEP] ' % (slot['slot'],slot['value'])
                                            knowledge = knowledge + temp
                                    except:
                                        nothing = 1
                                    try:
                                        if len(knowledge) == 0:
                                            knowledge = '[KNOWLEDGE] '
                                        try:
                                            intent = frame['state']['active_intent']
                                            temp = '%s [EQUAL] %s [SEP] ' % ('active_intent',intent)
                                            knowledge = knowledge + temp
                                            slot_values = frame['state']['slot_values']
                                            for slot in slot_values:
                                                vals = slot_values[slot]
                                                for val in vals:
                                                    temp = '%s [EQUAL] %s [SEP] ' % (slot,val)
                                                    knowledge = knowledge + temp
                                        except:
                                            nothing = 1
                                    except:
                                        noting = 1
                            
                            if len(prev_speaker) > 0:
                                if not bKnowledge:
                                    knowledge = ''
                                if dataset == 0:
                                    text = '[%s] %s %s [%s] %s [END]' % (prev_speaker,
                                                                         prev_utterance,
                                                                         knowledge,
                                                                         speaker,
                                                                         utterance)
                                else:
                                    text = '[%s] %s %s [%s] | %s [END]' % (prev_speaker,
                                                                         prev_utterance,
                                                                         knowledge,
                                                                         speaker,
                                                                         utterance)
                                fout.write('%s\n' % (text))
                                print(text)
                            prev_speaker = speaker
                            prev_utterance = utterance
    counts.append(count)
    count = 0
    print(counts)


def main():
    import pandas as pd
    gen_mode = 0
    gen_labels = ['logits','greedy','beam','top-p', 'top-k', 'temperature']
    tuned_model = 1
    
    if tuned_model == 0:
        tuned = 'gpt2'
        test_name = 'woz.test_a.txt'
    elif tuned_model == 1:
        tuned = '/content/gdrive/MyDrive/nlp/b'
        test_name = 'woz.test_b.txt'
    else:
        tuned = '/content/gdrive/MyDrive/nlp/c'
        test_name = 'woz.test_c.txt'
    
    tokenizer = GPT2Tokenizer.from_pretrained(tuned)
    model = GPT2LMHeadModel.from_pretrained(tuned,pad_token_id=tokenizer.eos_token_id)
    model = model.cuda()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])        
    print('total parameters = ',params)

    # Metrics
    metric_bleu = load_metric("bleu")
    
    print(metric_bleu)


    # # NEW METRICS ADDED
    metric_rouge = load_metric("rouge")
    metric_bleurt = load_metric("bleurt")
    # # metric_wikisplit = load_metric("wiki_split")
    
    predicts = []
    refs = []
    best = []
    bleus, bleu_best = [],[]
    # New metrics lists
    rouges, rouge_best = [],[]
    bleurts, bleurts_best = [],[]
    wikisplit = []

    max_len = 0
    total = 0
    with open(test_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            text = line.split('|')
            prompt = text[0].strip(' ')
            in_ids = tokenizer.encode(prompt,add_special_tokens=True)
            if len(in_ids) > max_len:
                max_len = len(in_ids)
            total = total + 1
            
    max_len = max_len + 32
    print('max_len: %d total: %d' % (max_len,total))
    
    obs = 0
    with open(test_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            text = line.split('|')
            prompt = text[0].strip(' ')
            ref = text[1].strip(' ')
            obs = obs + 1
            in_ids = tokenizer.encode(prompt,add_special_tokens=True)
            
            # Logits
            if gen_mode == 0:
                seq_len = 0
                bDone = False
                while not bDone:
                    input_ids = torch.tensor(in_ids).unsqueeze(0)
                    input_ids = input_ids.cuda()
                    outputs = model(input_ids, labels=input_ids)
                    decoded = []
                    for i in range(outputs[1].size(1)):
                        decoded.append(torch.argmax(outputs[1][0][i][:]).item())
                    decoded = torch.tensor(decoded)
                    decoded = decoded.cuda()
                    in_ids.append(decoded[decoded.size(0)-1].item())
                    input_ids = torch.tensor(in_ids).unsqueeze(0)
                    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    tokens = text.split(' ')
                    if tokens[len(tokens)-1] == '[END]':
                        bDone = True
                    if len(tokens) >= max_len:
                        bDone = True
            # Greedy decoder            
            if gen_mode == 1:
                input_ids = torch.tensor(in_ids).unsqueeze(0)
                input_ids = input_ids.cuda()
                greedy = model.generate(input_ids, max_length=max_len)  
                text2 = tokenizer.decode(greedy[0], skip_special_tokens=False)
                tokens = text2.split()
            
            # Beam search decoder    
            if gen_mode == 2:
                input_ids = torch.tensor(in_ids).unsqueeze(0)
                input_ids = input_ids.cuda()
                beam = model.generate(input_ids, max_length=max_len, num_beams = 5, early_stopping = True)  
                text2 = tokenizer.decode(beam[0], skip_special_tokens=False)
                tokens = text2.split()
            
            # Top-p decoder    
            if gen_mode == 3:
                input_ids = torch.tensor(in_ids).unsqueeze(0)
                input_ids = input_ids.cuda()
                top_p = model.generate(input_ids, max_length=max_len, do_sample = True, top_p = 0.90, top_k = 0)  
                text2 = tokenizer.decode(top_p[0], skip_special_tokens=False)
                tokens = text2.split()
            
            # NEW DECODERS ADDED
            # Top-k decoder
            if gen_mode == 4:
                input_ids = torch.tensor(in_ids).unsqueeze(0)
                input_ids = input_ids.cuda()
                top_k = model.generate(input_ids, max_length=max_len, do_sample = True, top_k = 50)  
                text2 = tokenizer.decode(top_k[0], skip_special_tokens=False)
                tokens = text2.split()
            
            # Temperature decoder
            if gen_mode == 5:
                input_ids = torch.tensor(in_ids).unsqueeze(0)
                input_ids = input_ids.cuda()
                temp = model.generate(input_ids, max_length=max_len, do_sample = True, top_k = 0, temperature=0.7)  
                text2 = tokenizer.decode(temp[0], skip_special_tokens=False)
                tokens = text2.split()

                
            first = len(prompt.split(' '))
            try:
                pos_end = tokens.index('[END]')
            except:
                pos_end = len(tokens)
            try:
                pos_enduser = tokens.index('[END][USER]')
            except:
                pos_enduser = len(tokens)
            try:
                pos_endsystem = tokens.index('[END][SYSTEM]')
            except:
                pos_endsystem = len(tokens)
            last = min(pos_end,pos_enduser,pos_endsystem,len(tokens))
            predict = ' '.join(tokens[first:last])

            predictions = [predict.split()]
            references = [[ref.split()]]
            predicts.append(predictions)
            refs.append(references)

     
            try:
                
                results_rouge = metric_rouge.compute(predictions=predictions, references=references)

                rouges.append(results_rouge['rougeLsum'].high.fmeasure)
                
                print('ref:  ', ref)
                print('pred: ', predict)
                
                if results_rouge['rougeLsum'].high.fmeasure > 0.01:
                    # print('ref:  ',ref)
                    # print('pred: ',predict)
                    print('Rouge[%d]: %7.3f' % (obs,results_rouge['rougeLsum'].high.fmeasure))
                    print(' ')
                    rouge_best.append(results_rouge['rougeLsum'].high.fmeasure)
                
               
            except Exception as e:
                print(e)
                continue
        if len(bleu_best)==0:
          bleus_avg = 0 
        else:
          bleus_avg = sum(bleu_best) / float(len(bleu_best))      
        
        print(rouge_best)     
        rouge_avg = sum(rouge_best) / float(len(rouge_best))
        print("r_avg",rouge_avg)
        print("b_aveg",bleus_avg)
        
        with open("prelim_results.txt", 'w') as write_file:
          write_file.write(f"\tAverage bests[BLEU=[{bleus_avg},  ROUGE={rouge_avg}]\n")

           
if __name__ == "__main__":
    main()