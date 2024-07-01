import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from collections import Counter
from transformers import PreTrainedModel
from transformers import PretrainedConfig
from transformers.models.bert.configuration_bert import BertConfig

class Model(nn.Module):
    def __init__(self, model, tokenizer, device=torch.device('cpu'), hidden_dim=768, num_labels=4):
        super().__init__()
        #super(Model, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hidden_dim = hidden_dim
        self.embedding_linear = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 4, num_labels))

    def forward(self, text_inputs):
        inputs = self.tokenizer(text_inputs, return_tensors="pt", padding='longest', add_special_tokens=True).to(self.device)
        embedding = self.model(**inputs).pooler_output
        outputs = self.projection_head(embedding)

        embeddig_out = torch.relu(self.embedding_linear(embedding))
        return outputs, embeddig_out





class EnsembleModel(nn.Module):
    def __init__(self, m1, m2, m3, pred_mode='combine'):
        super(EnsembleModel, self).__init__()
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.pred_mode = pred_mode

    def forward(self, text_inputs):
        outputs_m1, _ = self.m1(text_inputs)
        outputs_m2, _ = self.m2(text_inputs)
        outputs_m3, _ = self.m3(text_inputs)
        predicted_class1 = F.softmax(outputs_m1, dim=1)
        predicted_class2 = F.softmax(outputs_m2, dim=1)
        predicted_class3 = F.softmax(outputs_m3, dim=1)

        # Option 1 use the probabilities
        if self.pred_mode == 'combine':
            predicted_class = predicted_class1 + predicted_class2 + predicted_class3
            predicted_class = F.softmax(predicted_class, dim=1)
            predicted_class = torch.argmax(predicted_class, dim=1)

        elif self.pred_mode == 'voting':
            predicted_class1 = torch.argmax(predicted_class1, dim=1)
            predicted_class2 = torch.argmax(predicted_class2, dim=1)
            predicted_class3 = torch.argmax(predicted_class3, dim=1)
            print(predicted_class1, predicted_class2, predicted_class3)
            final_voting = []
            for i in range(predicted_class1.shape[0]):
                all_selections = Counter([predicted_class1[i].item(), predicted_class2[i].item(), predicted_class3[i].item()])
                selected_relation = all_selections.most_common()
                max_val = max([elem[1] for elem in selected_relation])
                selected_relation_aux = []
                for elem in selected_relation:
                    if elem[1] == max_val:
                        selected_relation_aux.append(elem[0])

                if len(selected_relation_aux) > 1:
                    if 2 in selected_relation_aux:
                        selected_relation = 2
                    elif 3 in selected_relation_aux:
                        selected_relation = 3
                    else:
                        selected_relation = selected_relation_aux[0]
                else:
                    selected_relation = selected_relation_aux[0]
                final_voting.append(selected_relation)

            predicted_class = torch.tensor(final_voting)

        else:
            predicted_class1 = torch.argmax(predicted_class1, dim=1)
            predicted_class2 = torch.argmax(predicted_class2, dim=1)
            predicted_class3 = torch.argmax(predicted_class3, dim=1)
            final_voting = []
            for i in range(predicted_class1.shape[0]):
                all_selections = [predicted_class1[i].item(), predicted_class2[i].item(), predicted_class3[i].item()]
                if all_selections[0] == all_selections[1] and all_selections[1] == all_selections[2]:
                    final_voting.append(all_selections[0])
                else:
                    final_voting.append(-1)

            predicted_class = torch.tensor(final_voting)

        return predicted_class



class EnsembleModel5(nn.Module):
    def __init__(self, m1, m2, m3, m4, m5, pred_mode='combine'):
        super(EnsembleModel5, self).__init__()
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.m4 = m4
        self.m5 = m5
        self.pred_mode = pred_mode

    def forward(self, text_inputs):
        outputs_m1, _ = self.m1(text_inputs)
        outputs_m2, _ = self.m2(text_inputs)
        outputs_m3, _ = self.m3(text_inputs)
        outputs_m4, _ = self.m4(text_inputs)
        outputs_m5, _ = self.m5(text_inputs)
        predicted_class1 = F.softmax(outputs_m1, dim=1)
        predicted_class2 = F.softmax(outputs_m2, dim=1)
        predicted_class3 = F.softmax(outputs_m3, dim=1)
        predicted_class4 = F.softmax(outputs_m4, dim=1)
        predicted_class5 = F.softmax(outputs_m5, dim=1)

        # Option 1 use the probabilities
        if self.pred_mode == 'combine':
            predicted_class = predicted_class1 + predicted_class2 + predicted_class3 + predicted_class4 + predicted_class5
            predicted_class = F.softmax(predicted_class, dim=1)
            predicted_class = torch.argmax(predicted_class, dim=1)

        elif self.pred_mode == 'voting':
            predicted_class1 = torch.argmax(predicted_class1, dim=1)
            predicted_class2 = torch.argmax(predicted_class2, dim=1)
            predicted_class3 = torch.argmax(predicted_class3, dim=1)
            predicted_class4 = torch.argmax(predicted_class4, dim=1)
            predicted_class5 = torch.argmax(predicted_class5, dim=1)

            final_voting = []
            for i in range(predicted_class1.shape[0]):
                all_selections = Counter([predicted_class1[i].item(), predicted_class2[i].item(), predicted_class3[i].item(),
                                          predicted_class4[i].item(),  predicted_class5[i].item()])
                selected_relation = all_selections.most_common()
                max_val = max([elem[1] for elem in selected_relation])
                selected_relation_aux = []
                for elem in selected_relation:
                    if elem[1] == max_val:
                        selected_relation_aux.append(elem[0])

                if len(selected_relation_aux) > 1:
                    if 2 in selected_relation_aux:
                        selected_relation = 2
                    elif 3 in selected_relation_aux:
                        selected_relation = 3
                    else:
                        selected_relation = selected_relation_aux[0]
                else:
                    selected_relation = selected_relation_aux[0]
                final_voting.append(selected_relation)

            predicted_class = torch.tensor(final_voting)

        else:
            predicted_class1 = torch.argmax(predicted_class1, dim=1)
            predicted_class2 = torch.argmax(predicted_class2, dim=1)
            predicted_class3 = torch.argmax(predicted_class3, dim=1)
            predicted_class4 = torch.argmax(predicted_class4, dim=1)
            predicted_class5 = torch.argmax(predicted_class5, dim=1)
            final_voting = []
            for i in range(predicted_class1.shape[0]):
                all_selections = [predicted_class1[i].item(), predicted_class2[i].item(), predicted_class3[i].item(),
                                  predicted_class4[i].item(), predicted_class5[i].item()]
                if all_selections[0] == all_selections[1] and all_selections[1] == all_selections[2]:
                    final_voting.append(all_selections[0])
                else:
                    final_voting.append(-1)

            predicted_class = torch.tensor(final_voting)

        return predicted_class



if __name__ == '__main__':
    model_to_finetune = 'michiyasunaga/BioLinkBERT-base'
    model_name = model_to_finetune.split('/')[1]
    device = torch.device('cuda')

    model = AutoModel.from_pretrained(model_to_finetune)
    tokenizer = AutoTokenizer.from_pretrained(model_to_finetune)

    mymodel = Model(model, tokenizer, device=device).to(device)

    text_inputs = ['holaaa [SEP] hahaha', 'chaaao']

    outputs = mymodel(text_inputs)
