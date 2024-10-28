import fire
# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
import torch
from transformers import pipeline
import pandas as pd
from ast import literal_eval
from tqdm.auto import tqdm
from vllm import LLM
from transformers import LlamaTokenizer
from collections import defaultdict
from WikiSER_relation_extraction import define_hardware,define_software,define_vulnerability,infer

from .inference import inference

def get_software(text_list):
    model_path='/home/hadi/Purdue/NER/universal-ner/src/train/saved_models/software-hf'
    llm = LLM(model=model_path, tensor_parallel_size=1)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    nlp=define_software()
    res_list=[]
    for sentences in text_list:
        entities=[]
        entity_types=['Application', 'Library', 'Operating_System']
        # for i in df.itertuples():
        #     entities.append(literal_eval(i[2]))
        #     entity_types.append(literal_eval(i[3]))
        

        res=[]
        for text in tqdm(sentences):
            res_entities=[]
            set_entities=set(entity_types)
            for entity_type in set_entities:
                if len(tokenizer(text + entity_type)['input_ids']) > 2048:
                    print(f"Error: Input is too long. Maximum number of tokens for input and entity type is {2048} tokens.")

                #preppend stuff
                #entity_type= "Software_Engineering_"+entity_type
                
                examples = [{"conversations": [{"from": "human", "value": f"Text: {text}"}, {"from": "gpt", "value": "I've read this text."}, {"from": "human", "value": f"What describes {entity_type} in the text?"}, {"from": "gpt", "value": "[]"}]}]

                output = inference(llm, examples, max_new_tokens=2048)[0]
                    #print(outputs[0]['generated_text'])
                
                if output:
                    #print(output)
                    try:
                        #print(literal_eval(output))
                        res_entities.extend(literal_eval(output))
                    except Exception as e:
                        #print(e)
                        #print(output)
                        pass
                        #res_entities.extend(output)
                    
            res_entities.extend(infer(nlp,text))
            #print(infer(nlp,text))
            res.append(res_entities)
        res_list.append(res)    
    return res_list

def get_hardware(text_list):
    model_path='/home/hadi/Purdue/NER/universal-ner/src/train/saved_models/hardware-hf'
    llm = LLM(model=model_path, tensor_parallel_size=1)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    nlp=define_software()
    res_list=[]
    for sentences in text_list:
        # text_list=df['Text'].to_list()
        entities=[]
        entity_types=[['Device']]
        res=[]
        for text,entity_types_list in tqdm(zip(sentences,entity_types)):
            res_entities=[]
            set_entities=set(entity_types_list)
            for entity_type in set_entities:
                if len(tokenizer(text + entity_type)['input_ids']) > 2048:
                    print(f"Error: Input is too long. Maximum number of tokens for input and entity type is {2048} tokens.")

                #preppend stuff
                #entity_type= "Software_Engineering_"+entity_type
                
                examples = [{"conversations": [{"from": "human", "value": f"Text: {text}"}, {"from": "gpt", "value": "I've read this text."}, {"from": "human", "value": f"What describes {entity_type} in the text?"}, {"from": "gpt", "value": "[]"}]}]
                output = inference(llm, examples, max_new_tokens=2048)[0]
                    #print(outputs[0]['generated_text'])
                
                if output:
                    try:
                        res_entities.extend(literal_eval(output))
                    except Exception as e:
                        print(e)
                        res_entities.extend(output)
                    
            res_entities.extend(infer(nlp,text))
            #print(res_entities)
            res.append(res_entities)
        res_list.append(res)
    return res_list

def get_vulnerability(text_list):
    
    res_list=[]
    for sentences in text_list:
        entity_types=[['Software_Weakness', 'Hardware_Weakness']]
        # for i in df.itertuples():
        #     entities.append(literal_eval(i[2]))
        #     entity_types.append(literal_eval(i[3]))
        
        model_path='/home/hadi/Purdue/NER/universal-ner/src/train/saved_models/vulnerability-hf'
        llm = LLM(model=model_path, tensor_parallel_size=1)
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        nlp=define_software()
        res=[]
        for text,entity_types_list in tqdm(zip(sentences,entity_types)):
            res_entities=[]
            set_entities=set(entity_types_list)
            for entity_type in set_entities:
                if len(tokenizer(text + entity_type)['input_ids']) > 2048:
                    print(f"Error: Input is too long. Maximum number of tokens for input and entity type is {2048} tokens.")

                #preppend stuff
                #entity_type= "Software_Engineering_"+entity_type
                
                examples = [{"conversations": [{"from": "human", "value": f"Text: {text}"}, {"from": "gpt", "value": "I've read this text."}, {"from": "human", "value": f"What describes {entity_type} in the text?"}, {"from": "gpt", "value": "[]"}]}]
                output = inference(llm, examples, max_new_tokens=2048)[0]
                    #print(outputs[0]['generated_text'])
                
                if output:
                    try:
                        print(literal_eval(output))
                        res_entities.extend(literal_eval(output))
                    except Exception as e:
                        print(e)
                        res_entities.extend(output)
                    
            res_entities.extend(infer(nlp,text))
            #print(res_entities)
            #set_entities=list(set(item for sublist in res_entities for item in sublist))
            res.append(res_entities)
        res_list.append(res)
    return res_list


def main(

):      
    df=pd.read_csv('Hacker_100.csv')
    
    text_list=[]

    for items in df.itertuples():
        #print(items[1])
        text_list.append(literal_eval(items[1]))
    
    # text_list=[text_list[2]]
    software_list=get_software(text_list)
    #print(software_list)
    # hardware_list=get_hardware(text_list)
    # vulnerability_list=get_vulnerability(text_list)
    
    
    df['Software']=software_list
    # # df['Hardware']=hardware_list
    # # df['Vulnerability']=vulnerability_list
    
    
    df.to_csv('Hacker_100.csv', index=False)

if __name__ == "__main__":
    fire.Fire(main)