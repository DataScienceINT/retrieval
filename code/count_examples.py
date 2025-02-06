from prodigy.components.db import connect

db = connect()
all_dataset_names = db.datasets
examples = db.get_dataset_examples("kidney-retrieval-annotations")

def get_count(annotator, examples):
    c = sum([annotator in example["_annotator_id"]for example in examples])
    ret = annotator + ": " + str(c) + " annotations"
    return ret

print(get_count("devon", examples))
print(get_count("kate", examples))