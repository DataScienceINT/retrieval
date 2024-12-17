import prodigy
import csv
import random
#from prodigy.components.preprocess import add_tokens
from prodigy.components.stream import Stream
from prodigy.util import set_hashes
import spacy
import copy
from prodigy.components.routers import full_overlap

@prodigy.recipe(
	"retrieval-validation",
	dataset=("The dataset to use", "positional", None, str),
    	spacy_model=("The base model", "positional", None, str))
def retrieval(dataset, source, spacy_model):
    # We can use the blocks to override certain config and content, and set
    # "text": None for the choice interface so it doesn't also render the text
    blocks = [
        {"view_id": "ner_manual"},
        {"view_id": "choice"},
        {"view_id": "text_input", "field_rows": 3, "field_label": "If ambiguous, why?"}
    ]
    options = [
        {"id": 2, "text": "😺 Relevant"},
        {"id": 1, "text": "🙀 Ambiguous"},
        {"id": 0, "text": "😾 Not relevant"}
    ]


    def custom_csv_loader(source, encoding='utf-8-sig'): 
        #examples = list()
        with open(source) as csvfile: 
            reader = csv.DictReader(csvfile)
            rows = list(reader)  # Load all rows into a list
            random.shuffle(rows)  # Shuffle the rows 
            for row in rows: 
                r_score = row.get('relevance score') 
                text = row.get('query') + '\n\n' + row.get('passage text')
                s_score = row.get('similarity score')
                method = row.get('method')
                meta = row.get('metadata')
                row_type = row.get('type')
                yield {'text': text, "options": options,'rel_score':r_score,'sim_score':s_score, 'metadata': meta, 'method': method, 'type':row_type}
        #return examples

    def make_tasks(nlp, stream): #, labels
        #from ner.correct code
        """Add a 'spans' key to each example, with predicted entities."""
        # Process the stream using spaCy's nlp.pipe, which yields doc objects.
        # If as_tuples=True is set, you can pass in (text, context) tuples.
        texts = ((eg["text"], eg) for eg in stream)
        for doc, eg in nlp.pipe(texts, as_tuples=True):
            task = copy.deepcopy(eg)
            spans = []
            for ent in doc.ents:
                # Create a span dict for the predicted entity.
                spans.append(
                    {
                        "token_start": ent.start,
                        "token_end": ent.end - 1,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "text": ent.text,
                        "label": ent.label_,
                    }
                )
            task["spans"] = spans
            # Rehash the newly created task so that hashes reflect added data.
            task = set_hashes(task)
            yield task
        
    stream_as_generator = custom_csv_loader(source) #load stream with custom loader
    stream = Stream.from_iterable(stream_as_generator) # convert it into Stream
    #if spacy_model is None:
    #    nlp=spacy.load("en_blank")
    #else:
    nlp = spacy.load(spacy_model)
    #stream.apply(add_tokens, nlp=nlp, stream=stream)  # tokenize the stream for ner_manual
    stream.apply(make_tasks, nlp=nlp, stream=stream) 
    #TODO:hierarchy for query as title
    return {
        "dataset": dataset,          # the dataset to save annotations to
        "view_id": "blocks",         # set the view_id to "blocks"
        "stream": stream,            # the stream of incoming examples
        #"task_router": full_overlap, # all annotators see all examples
        #"annotations_per_task": 2,
        "config": {
            "labels": [""],
            "blocks": blocks,
	    #"custom_theme": {
            #    "cardMaxWidth": "95%"
    	    #	}         # add the blocks to the config
        }
    }


