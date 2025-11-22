import spacy
from conllu import parse_incr
from pathlib import Path

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 3000000 # not a smart solution
input_folder =Path("./data/conllu")
output_folder = Path("./spacy_results")
output_folder.mkdir(exist_ok=True)

TARGETS = {"PERSON","ORG","GPE"}

# gets plaintext from conllu format for pretrained spacy
def get_plain_from_conllu(file):
    print(file)
    plaintext=[]
   
    with open(file,"r",encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("# text ="):
                plaintext.append(line[len("# text ="):])
        
    return plaintext

# does the named entity recognition on one file       
def entity_recognition(doc):
    results = []
    word_entity_list = [f"{ent.text}:{ent.label_}" for ent in doc.ents if ent.label_ in TARGETS]
    if word_entity_list:  # only keep sentences with relevant entities
        results.append({
            "entities": word_entity_list
        })
    return results
    


# foreach file pass it to the plaintext function, get the named entities, and save to a txt file
for file_path in input_folder.glob("*.conllu"):
    out_file = output_folder / f"{file_path.stem}.json"
    if out_file.exists():
        continue
    
    plain_text = get_plain_from_conllu(file_path)
    struct_data = entity_recognition(nlp(str(plain_text)))
    with open(out_file,"w",encoding="utf-8") as f:
        f.write(str(struct_data))




