import spacy
from pathlib import Path

# supports conll but not conllu
#from spacy_conll import init_parser
#from spacy_conll.parser import ConllParser
#nlp = ConllParser(init_parser("de_core_news_md", "spacy"))

nlp = spacy.load("de_core_news_md")
nlp.max_length = 3000000 # not a smart solution but it works
input_folder =Path("./data/test_set_txt")
output_folder = Path("./spacy_results")
output_folder.mkdir(exist_ok=True)

TARGETS = {"PER","ORG","LOC"}

# gets plaintext from conllu format for pretrained spacy - deprecated
"""def get_plain_from_conllu(file):
    print(file)
    plaintext=[]
   
    with open(file,"r",encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("# text ="):
                plaintext.append(line[len("# text ="):])
        
    return plaintext"""

# does the named entity recognition on one file       
def entity_recognition(doc):
    results = []
    word_entity_list = [f"{ent.text}:{ent.label_}" for ent in doc.ents if ent.label_ in TARGETS]
    if word_entity_list:  # only keep sentences with relevant entities
        results.append({
            "sent":doc.text,
            "entities": word_entity_list
        })
    return results
    


# foreach file pass it to the plaintext function, get the named entities, and save to a txt file
for file_path in input_folder.glob("*.txt"):
    out_file = output_folder / f"{file_path.stem}.txt"
    if out_file.exists():
        continue
    
    # could in theory be done from the conllu file but
    # just using the plain text is more convenient
    # doc = nlp.parse_conll_file_as_spacy(file_path)

    #plain_text = get_plain_from_conllu(file_path)
    with open(file_path,"r",encoding="utf-8") as f:
        plain_text = f.read()

    struct_data = entity_recognition(nlp(str(plain_text)))
 
    with open(out_file,"w",encoding="utf-8") as f:
        f.write(str(struct_data))




