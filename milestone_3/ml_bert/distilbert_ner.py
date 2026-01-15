from transformers import pipeline
from ..entities import Entity
import torch

# Use GPU if available (0), otherwise CPU (-1)
DEVICE = 0 if torch.cuda.is_available() else -1
MODEL_NAME = "Davlan/distilbert-base-multilingual-cased-ner-hrl"

TARGETS = {"PER", "ORG"}

class DistilBertNer:
    def __init__(self):
        self.pipe = pipeline(
            "ner", 
            model=MODEL_NAME, 
            tokenizer=MODEL_NAME, 
            aggregation_strategy="simple",
            device=DEVICE
        )

    def annotate(self, text: str) -> list[Entity]:
        entities: list[Entity] = []
        
        # read text in 2000 char blocks - chunking
        window_size = 2000 
        overlap = 100
        text_len = len(text)
        start_idx = 0
        seen_spans = set()

        while start_idx < text_len:
            end_idx = min(start_idx + window_size, text_len)
            chunk = text[start_idx:end_idx]
            
            try:
                results = self.pipe(chunk)
                
                for ent in results:
                    label = ent['entity_group']
                    if label in TARGETS:
                        real_start = ent['start'] + start_idx
                        real_end = ent['end'] + start_idx
                        
                        span_id = (real_start, real_end, label)
                        
                        if span_id not in seen_spans:
                            entities.append(Entity(
                                text=ent['word'],
                                label=label,
                                start=real_start,
                                end=real_end
                            ))
                            seen_spans.add(span_id)
            except Exception as e:
                print(f"Warning: DistilBERT failed on chunk. Error: {e}")

            if start_idx + window_size >= text_len:
                break
            
            start_idx += (window_size - overlap)

        return entities