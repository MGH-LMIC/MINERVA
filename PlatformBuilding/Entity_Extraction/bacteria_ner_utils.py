import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from transformers import AutoTokenizer, DistilBertForTokenClassification
import torch
import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DistilbertNER(nn.Module):
    """
    Implement NN class based on distilbert pretrained from Hugging face.
    Inputs :
      tokens_dim : int specifyng the dimension of the classifier
    """

    def __init__(self, tokens_dim):
        super(DistilbertNER, self).__init__()

        if type(tokens_dim) != int:
            raise TypeError('Please tokens_dim should be an integer')

        if tokens_dim <= 0:
            raise ValueError('Classification layer dimension should be at least 1')

        self.pretrained = DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased",
                                                                           num_labels=tokens_dim)  # set the output of each token classifier = unique_lables

    def forward(self, input_ids, attention_mask, labels=None):  # labels are needed in order to compute the loss
        """
      Forwad computation of the network
      Input:
        - inputs_ids : from model tokenizer
        - attention :  mask from model tokenizer
        - labels : if given the model is able to return the loss value
      """

        # inference time no labels
        if labels == None:
            out = self.pretrained(input_ids=input_ids, attention_mask=attention_mask)
            return out

        out = self.pretrained(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return out


def tags_mapping(tags_series: pd.Series):
    """
    tag_series = df column with tags for each sentence.
    Returns:
      - dictionary mapping tags to indexes (label)
      - dictionary mappign inedexes to tags
      - The label corresponding to tag 'O'
      - A set of unique tags ecountered in the trainind df, this will define the classifier dimension
    """

    if not isinstance(tags_series, pd.Series):
        raise TypeError('Input should be a pandas Series')

    unique_tags = set()

    for tag_list in tags_series:
        for tag in tag_list.split():
            unique_tags.add(tag)

    tag2idx = {k: v for v, k in enumerate(sorted(unique_tags))}
    idx2tag = {k: v for v, k in tag2idx.items()}

    unseen_label = tag2idx["O"]

    return tag2idx, idx2tag, unseen_label, unique_tags




def combine_tokens(tokens, word_ids, predictions, NER):
    """
    takes tokens, word_ids and predictions as input
    combines tokens into whole words using word ids
    combines predictions for tokens based on word ids

    returns a dictionary with 'B' as bacteria and 'D' as disease
    """
    grouped_tokens = {}
    for token, label in zip(tokens, word_ids):
        if label not in grouped_tokens:
            grouped_tokens[label] = []
        grouped_tokens[label].append(token)
        ''.join(grouped_tokens[label])

    grouped_preds = {}

    for label, pred in zip(word_ids, predictions):
        if label not in grouped_preds:
            grouped_preds[label] = []
        grouped_preds[label].append(int(pred))

    for key in list(grouped_tokens.keys()):
        grouped_tokens[key] = ''.join(grouped_tokens[key])

    df = pd.DataFrame.from_dict(grouped_tokens, orient='index', columns=['words'])

    preds = []
    for v in list(grouped_preds.values()):
        preds.append(math.ceil(sum(v) / len(v)))

    df['preds'] = preds

    filtered = df[df['preds'] != 1]
    filtered = filtered.reset_index()

    output = {NER: []}

    for i in range(len(filtered)):
        word = ''.join(filtered['words'][i].split('##'))
        value = filtered['preds'][i]
        if value == 0:
            output[NER].append(word)

    return output


def get_prediction(model, tokenizer, sentence, NER='D'):
    """
    -takes a sentence as input
    -creates a list of tokens and a list of word_ids
    -gets model predictions
    -inputs the 3 lists to combine_tokens()
    -returns combine_tokens() output
    """
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512, padding=True)
    word_ids = inputs.word_ids()
    tokens = tokenizer.tokenize(sentence)
    tokens.insert(0, 'none')
    tokens.append('none')

    mask = inputs['attention_mask'].squeeze(1).to(device)
    input_id = inputs['input_ids'].squeeze(1).to(device)

    output = model(input_id, mask)[0]

    predictions = output.argmax(-1)[0].cpu().numpy().tolist()

    return combine_tokens(tokens, word_ids, predictions,
                          NER)  # if you are predicting bacteria NER,please input 'B' as NER argument.


if __name__ == '__main__':
    # df_test = pd.read_csv('../../../databases/llms-microbe-disease/data/gold_data_corrected.csv')#pd.read_csv('disease_NER_silver_D.csv')
    # evidence_groups = df_test.groupby('EVIDENCE')
    # evidences = []
    # microbes = []
    # diseases = []
    # for evidence, df_group in evidence_groups:
    #     #print(evidence)
    #     #print(df_group['MICROBE'].values, df_group['DISEASE'].values)
    #     evidences.append(evidence)
    #     microbes.append(list(set([elem.lower() for elem in df_group['MICROBE'].values])))
    #     diseases.append(list(set([elem.lower() for elem in df_group['DISEASE'].values])))


    # Test disease
    # tag2idx, idx2tag, unseen_label, unique_tags = tags_mapping(pd.read_csv('disease_NER_silver_D.csv')["labels"])
    #
    # device = torch.device('cpu')  # ("cuda" if torch.cuda.is_available() else "cpu")
    # model = DistilbertNER(len(unique_tags))
    # model_path = 'disease_distillBERT_silver_20_epochs_D.pt'
    # model = torch.load(model_path, map_location='cpu').to(device)
    # tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    #
    # for i in range(len(evidences)):
    #     prediction = get_prediction(evidences[i], NER='D')
    #     print('Prediction: {} | Real: {}'.format(prediction['D'], diseases[i]))

    # Test disease
    #tag2idx, idx2tag, unseen_label, unique_tags = tags_mapping(pd.read_csv('bacteria_NER_silver_B.csv')["labels"])
    #print(unique_tags)
    device = torch.device('cpu')  # ("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilbertNER(len({'O', 'B'}))
    model_path = 'bacteria_distillBERT_silver_20_epochs_B_2.pt'
    model = torch.load(model_path, map_location='cpu').to(device)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    evidences = ['Eubacteria produces cancer']
    for i in range(len(evidences)):
        prediction = get_prediction(model=model, tokenizer=tokenizer, sentence=evidences[i], NER='B')
        print(prediction['B'])
        #print('Prediction: {} | Real: {}'.format(prediction['B'], microbes[i]))
