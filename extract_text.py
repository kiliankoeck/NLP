import re
import nltk
from nltk.tokenize import sent_tokenize
# start with "Beginn der Sitzung: x{x}.xx Uhr"
# end with "Schluss der Sitzung: x{x}.xx Uhr"

with open("steno_protocols/plain_text/XXV_BRSITZ_851.txt","r", encoding="utf-8") as f:
    document = f.read()

document = ' '.join(document.split()) # get it all in one line
document = re.sub(r"\s+([.,!?])", r"\1",document) # remove whitespaces  before .,!?

pattern = r"Beginn der Sitzung: \d{1,2}[.]\d{2} Uhr(.*?)Schluss der Sitzung: \d{1,2}[.]\d{2} Uhr"
match = re.search(pattern,document,re.DOTALL)
protocol = match.group(1).strip() if match else document

# remove additional remarks in parentheses (mostly referring to applause) 
# this also removes which parties the speakers are from (could be done at a later time?)
# and things shouted inbetween by other people
protocol = re.sub(r"\([^()]*\)","",protocol)

# remove the new page headers e.g. (Bundesrat Stenographisches Protokoll 851. Sitzung / Seite 10)
protocol = re.sub(r"Bundesrat Stenographisches Protokoll(.*?)Seite \d*","",protocol,re.DOTALL)
protocol = protocol.replace("\xad","") # remove soft hyphens


# protocol now looks like speaker:text ... time
# in the following, the protocol will be split into speaker:text as key value pairs
pattern = r"([A-ZÄÖÜ][a-zäöüßA-ZÄÖÜ]+)\s*:\s*(.*?) (?=\d{1,2}[.]\d{2}\d?|\bSchluss der Sitzung:|\Z)"
matches = re.findall(pattern, protocol, re.DOTALL)
results = [
    {'speaker': s.strip(), 'text': ' '.join(t.split())}
    for s, t in matches
]

for line in results:
    v = line['text']
    v = re.sub(r"\s{2,}", " ", v).strip() # remove double whitespaces
    # now we sentence tokenize the text
    line['tokenized'] = nltk.sent_tokenize(v, language="german")


# write the resulting information to a file
with open ("tmp.txt","w",encoding="utf-8") as f:
    for line in results:
        f.write(str(line['speaker'])+": " +str(line["tokenized"])+"\n")
