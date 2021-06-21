import spacy
import json
from spacy import displacy
from collections import Counter
import en_core_web_sm

nlp = en_core_web_sm.load()


def write_json(sentence):
    doc = nlp(sentence)
    my_list = []
    entities = [X.text for X in doc.ents]

    ent_dict = []
    for i in range(0, len(entities) - 1):
        for j in range(i + 1, len(entities)):
            ent_dict.append({"arg1": entities[i], "arg2": entities[j]})

    my_list.append({"text": sentence, "entities": ent_dict})
    return my_list