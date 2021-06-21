#!pip install transformers
import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import BertTokenizer, RobertaTokenizer, RobertaForSequenceClassification, AdamW, \
    BertForSequenceClassification
from transformers import DistilBertForTokenClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import statistics
import pickle
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


def predict_tag_sentence(tokenizer, model, text, tag, max_length=256):
    sequence = tokenizer.encode_plus(tag, text, max_length=max_length,
                                     padding='max_length', truncation="longest_first"
                                     , return_tensors="pt")['input_ids']

    logits = model(sequence)[0]
    probabilities = torch.softmax(logits, dim=1).detach().cpu().tolist()[0]
    if probabilities[1] > 0.5:
        return 1
    return 0


def evaluate(pred_labels, test_labels):
    pred_labels1 = np.array_split(pred_labels, 5)
    test_labels1 = np.array_split(test_labels, 5)
    accuracy = []
    f1 = []
    for test, pred in zip(test_labels1, pred_labels1):
        accuracy.append(accuracy_score(test, pred))
        f1.append(f1_score(test, pred, average="weighted"))

    print("Accuracy: " + str(sum(accuracy) / len(accuracy)))
    print("Standard Deviation: " + str(statistics.stdev(accuracy)))

    print("F1 Score: " + str(sum(f1) / len(f1)))
    print("Standard Deviation: " + str(statistics.stdev(f1)))


def read_data_tag_sentence(tags):
    context_tags = []

    for tagg in tags:
        context_tags.append(tagg[0])

    return context_tags


def tokenizeData_tag_sentence(tokenizer, text, tags, max_length=256):
    input_ids = []
    attention_masks = []

    for tx, tg in zip(text, tags):
        tokenizedData = tokenizer.encode_plus(tx, tg, max_length=max_length,
                                              padding='max_length', truncation="longest_first")
        tokenizedQP = tokenizedData["input_ids"]
        attentionMask = tokenizedData["attention_mask"]

        input_ids.append(tokenizedQP)
        attention_masks.append(attentionMask)

    return np.array(input_ids), np.array(attention_masks)


def read_labels(dataa):
    labels = []
    count = 0
    for dat in dataa:
        for ent in dat["entities"]:
            labels.append(int(ent["label"]))
            count += 1
    return labels


def bert_tag_sentence(data, sentence):
    text_list = []
    tag_list = []

    for d in data:
        text_list.append([d["text"].strip("\n")])
        for ent in d["entities"]:
            tag_list.append([d["text"].replace(ent["arg1"] + " ", "ENTITY1 ").replace(" " + ent["arg2"] + " ",
                                                                                      " ENTITY2 ").strip(
                "\n")])

    tags = read_data_tag_sentence(tag_list)
    # labels = read_labels(data)

    sentences = []
    args1 = []
    args2 = []

    for d in data:
        sentences.append(d["text"])
        for ent in d["entities"]:
            args1.append([ent["arg1"]])
            args2.append([ent["arg2"]])

    # train_texts, test_texts, train_tags, test_tags, train_labels, test_labels = train_test_split(text, tags,
    # labels, test_size=.2)

    # train_arg1, test_arg1, train_arg2, test_arg2 = train_test_split(args1, args2, test_size=.2)

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    # train_ids, train_attn = tokenizeData_tag_sentence(tokenizer, train_texts, train_tags)
    # test_ids, test_attn = tokenizeData_tag_sentence(tokenizer, test_texts, test_tags)

    # trainFeatures = (train_ids, train_attn, train_labels)
    # testFeatures = (test_ids, test_attn)

    # testDataLoader = buildDataLoaders(8, testFeatures)
    model = torch.load('data/trained_model.pkl')

    print(sentence)

    pred_labels = []
    for tag in tags:
        pred_labels.append(predict_tag_sentence(tokenizer, model, sentence, tag))

    # evaluate(pred_labels, test_labels)

    pred_args = []
    for arg1, arg2, label in zip(args1, args2, pred_labels):
        pred_args.append({"arg1": arg1[0], "arg2": arg2[0], "label": label})

    # with open('predictions.json', 'w') as f:
    #   json.dump(pred_args, f, indent=4)
    return pred_args


def execute(sentence):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    data = write_json(sentence)
    predictions = bert_tag_sentence(data, sentence)
    return predictions
