"""
Converts the JSON output of xmi_parser.py into proper CONLL-U format:
  - Sentence blocks preserved
  - Token start/end offsets in MISC
  - NER label column included

** input and output folders are for the test set
** need to change that for the full dataset
"""
import os
import json

def format_to_conllu(json_data, file_name="unknown"):
    tokens = json_data["tokens"]
    bio_tags = json_data["bio_tags"]
    sentences = json_data.get("sentences", [(0, len(tokens))])
    conllu_sentences = []

    for idx, (sent_start, sent_end) in enumerate(sentences, 1):
        # sentence header
        sent_text = "".join([tok[2] for tok in tokens[sent_start:sent_end]])
        header = f"# file = {file_name}\n# id = {idx}\n# text = {sent_text}"
        
        lines = []
        for i, (tok, tag) in enumerate(zip(tokens[sent_start:sent_end], bio_tags[sent_start:sent_end]), 1):
            begin, end, text = tok
            misc = f"start_char={begin}|end_char={end}|NER={tag}"
            lines.append(f"{i}\t{text}\t_\t_\t_\t_\t0\troot\t_\t{misc}")
        
        conllu_sentences.append(header + "\n" + "\n".join(lines))
    return "\n\n".join(conllu_sentences) + "\n"


def main(input_folder: str, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    json_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".json")]

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as fin:
            data = json.load(fin)
        conllu_text = format_to_conllu(data)
        out_file = os.path.join(output_folder, os.path.basename(jf).replace(".json", ".conllu"))
        with open(out_file, "w", encoding="utf-8") as fout:
            fout.write(conllu_text)
        print(f"Converted {jf} -> {out_file}")

if __name__ == "__main__":
    import sys
    #testset
    input_folder = "data/test_set_json/"
    output_folder = "data/test_set_conllu/"

    ##full dataset
    #input_folder = "data/raw_json/"
    #output_folder = "data/conllu/"
    main(input_folder, output_folder)
