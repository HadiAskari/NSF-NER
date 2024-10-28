# NSF-NER-RE


**Infering the NER Models**

The command to run the infer the NER Models can be found in the file UniNER/universal-ner/infer.sh . One needs to do the following changes:
- Change model path to one of the following in the infer.sh file: hadiaskari98/Vulnerability_UniNER, hadiaskari98/Hardware_UniNER, or hadiaskari98/Software_UniNER.
- Change the model/tokenizer path in the UniNER/universal-ner/WikiSER.py to one of the following: hadiaskari98/Vulnerability_NER_prod, hadiaskari98/Hardware_NER_prod, hadiaskari98/Software_NER_prod
- Change line 41 in UniNER/universal-ner/src/serve/cli_ser_for_demo.py to one of the following: for text,entity_types_list in tqdm(zip([text_new],[['HARDWARE']])):, for text,entity_types_list in tqdm(zip([text_new],[['SOFTWARE']])):, for text,entity_types_list in tqdm(zip([text_new],[['VULNERABILITY']])):

Sample command is given below:

```
python -m src.serve.cli_ser_for_demo --text_new "Software vulnerabilities, like buffer overflow and SQL injection flaw, can expose systems to severe security risks, allowing attackers to gain unauthorized access or execute malicious code." --model_path hadiaskari98/Vulnerability_UniNER  --tensor_parallel_size 1 --max_input_length 2048 --max_new_tokens 2048
                                  
```


**Relation Extraction**

The command to run the RE model is in Relation_Extraction/run.sh. The following the a sample command with an article and 3 lists of named entities relating to Software, Hardware and Vulnerabilities:

```
python -m script_llama3_hacker --text "The Microsoft Exchange Server has been a frequent target of cyber espionage campaigns, with zero-day vulnerabilities allowing attackers to compromise sensitive communication systems. Similarly, Fortinet FortiOS, the operating system for Fortinet security appliances, has faced vulnerabilities exploited by threat actors to bypass network protections, compromising connected hardware. These software and hardware platforms underscore the ongoing risks posed by unpatched vulnerabilities within critical enterprise environments." --software '["Microsoft Exchange Server", "Fortinet FortiOS"]' --hardware '["Fortinet security appliances"]' --vulnerability '["Zero-day vulnerabilities", "Unpatched vulnerabilities"]'
