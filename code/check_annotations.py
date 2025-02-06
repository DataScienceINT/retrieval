from prodigy.components.db import connect
from collections import Counter

db = connect()
all_dataset_names = db.datasets
examples = db.get_dataset_examples("kidney-retrieval-annotations")

relevance_mapping = {0: "Not relevant", 1: "Ambiguous", 2: "Relevant", 3: "Relevant but secondary evidence"}

def util_func(a): 
    #remove examples where relevance wasnt scored
    try: 
        return {"prompt":a['text'].split("\n")[0], "relevance":a["accept"][0]}
    except : 
        pass

prompts = [util_func(example) for example in examples]
prompts = [p for p in prompts if p is not None]
for p in prompts:
    p["relevance"] = relevance_mapping.get(p["relevance"], "Unknown")

counts = Counter((p["prompt"], p["relevance"]) for p in prompts)
print(counts)
with open("prompts_counts.csv", encoding='utf-8-sig', mode='w') as fp:
    fp.write('prompt,relevance,freq\n')  
    for prompt, count in counts.items():  
        fp.write('{},{},{}\n'.format(prompt[0], prompt[1], count))