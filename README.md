# NSF-NER

The command to run the RE model is in run.sh. The following the a sample command with an article and 3 lists of named entities relating to Software, Hardware and Vulnerabilities:

```python -m script_llama3_hacker --text "The Microsoft Exchange Server has been a frequent target of cyber espionage campaigns, with zero-day vulnerabilities allowing attackers to compromise sensitive communication systems. Similarly, Fortinet FortiOS, the operating system for Fortinet security appliances, has faced vulnerabilities exploited by threat actors to bypass network protections, compromising connected hardware. These software and hardware platforms underscore the ongoing risks posed by unpatched vulnerabilities within critical enterprise environments." --software '["Microsoft Exchange Server", "Fortinet FortiOS"]' --hardware '["Fortinet security appliances"]' --vulnerability '["Zero-day vulnerabilities", "Unpatched vulnerabilities"]'
