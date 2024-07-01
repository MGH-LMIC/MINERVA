import numpy as np
import torch
from new_release_paper.normal_db.data_loader import DataLoader as MyDataloader
from create_dataset import MyDataset
from transformers import AutoTokenizer, AutoModel
from model import Model, EnsembleModel
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import classification_report
from finetune import FinetuneBert
import os

device = torch.device('cuda')

def get_weights(dataset, dicto_classes):
    weights1 = dataset['RELATION'].value_counts()
    weights1 = dataset.shape[0] / weights1
    weights1 = torch.tensor([weights1[dicto_classes[0]], weights1[dicto_classes[1]],
                             weights1[dicto_classes[2]], weights1[dicto_classes[3]]])
    weights1 = weights1 / torch.sum(weights1)
    return weights1

class FinetuneBertEnsemble:
    def __init__(self, model, model_name='ensemble', encode_relations={}, folder=''):
        self.model = model
        self.model_name = model_name
        self.folder = folder
        self.encode_relations = encode_relations

    def test(self, test_dataloader, k=0):
        with torch.no_grad():
            all_preds = []
            all_targets = []
            for i_batch, sample_batched in enumerate(test_dataloader):
                inputs = sample_batched['inputs']
                relations = sample_batched['relations'].to(device)

                # Optimization loop
                predicted_class = self.model(inputs)
                all_targets.append(relations.cpu().numpy().squeeze())
                all_preds.append(predicted_class.cpu().numpy().squeeze())


            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)

            if self.model.pred_mode == 'agreement':
                all_targets_aux = []
                all_preds_aux = []
                for i in range(len(all_preds)):
                    if all_preds[i] == -1:
                        pass
                    else:
                        all_preds_aux.append(all_preds[i])
                        all_targets_aux.append(all_targets[i])
                all_preds = np.array(all_preds_aux)
                all_targets = np.array(all_targets_aux)


            report = classification_report(y_true=all_targets, y_pred=all_preds, labels=[0, 1, 2, 3],
                                           target_names=self.encode_relations.keys(), output_dict=True)
            report['total support'] = len(all_preds)
            print('{} Ensemble Report {} for k:{} {}'.format('#'*50, k, self.model.pred_mode, '#'*50))
            print(report)
            torch.save(report, '{}/{}_report_{}.pkl'.format(self.folder, self.model_name, self.model.pred_mode))


if __name__ == '__main__':
    use_gold = True
    split_gold = True
    batch_size = 32
    epochs = 100
    alpha_contrastive = 0
    local_model = True
    silver_model = False
    use_llm_augmentations = True
    use_llm_RAG_augmentations = False
    use_shuffling_augmentations = False
    use_weights = False
    #model_to_finetune = 'michiyasunaga/BioLinkBERT-base'
    model_to_finetune = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    #model_to_finetune = 'emilyalsentzer/Bio_ClinicalBERT'
    model_name = model_to_finetune.split('/')[1]


    if local_model:
        model_name = model_name + '_MLM_v2'
        my_model_to_finetune = 'mask_bert/results/best_bio_bert_v2'
    elif silver_model:
        model_name = model_name + '_SILVER'
        my_model_to_finetune = 'silver_bert/results/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext_SILVER_best.pt'

    if use_llm_augmentations:
        model_name += '_llmAUG'
    if use_shuffling_augmentations:
        model_name += '_shufflingAUG'
    if use_llm_RAG_augmentations:
        model_name += '_llmRAGAUG'

    if use_weights:
        model_name += '_Weighted'

    folder = 'results/{}/'.format(model_name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    device = torch.device('cuda')

    data_path = '../databases/llms-microbe-disease/data/gold_data_corrected.csv'

    data_loader1 = MyDataloader(data_path=data_path,
                                use_gold=use_gold, split_gold=split_gold,
                                 k=0)
    data_loader2 = MyDataloader(data_path=data_path,
                                use_gold=use_gold, split_gold=split_gold,k=1)
    data_loader3 = MyDataloader(data_path=data_path,
                                use_gold=use_gold, split_gold=split_gold,
                                 k=2)
    data_loader4 = MyDataloader(data_path=data_path,
                                use_gold=use_gold, split_gold=split_gold,
                                k=3)
    data_loader5 = MyDataloader(data_path=data_path,
                                use_gold=use_gold, split_gold=split_gold,
                                k=4)

    datasets1 = {0: {'train': pd.concat([data_loader3.data_augmented, data_loader4.data_augmented, data_loader5.data_augmented], axis=0),
                    'test': data_loader1.data, 'validation': data_loader2.data},
                1: {'train': pd.concat([data_loader1.data_augmented, data_loader4.data_augmented, data_loader5.data_augmented], axis=0),
                    'test': data_loader2.data, 'validation': data_loader3.data},
                2: {'train': pd.concat([data_loader1.data_augmented, data_loader2.data_augmented, data_loader5.data_augmented], axis=0),
                    'test': data_loader3.data, 'validation': data_loader4.data},
                3: {'train': pd.concat([data_loader1.data_augmented, data_loader2.data_augmented, data_loader3.data_augmented], axis=0),
                    'test': data_loader4.data, 'validation': data_loader5.data},
                4: {'train': pd.concat([data_loader2.data_augmented, data_loader3.data_augmented, data_loader4.data_augmented], axis=0),
                    'test': data_loader5.data, 'validation': data_loader1.data}}

    datasets2 = {0: {'train': pd.concat([data_loader2.data_augmented, data_loader4.data_augmented, data_loader5.data_augmented], axis=0),
                     'test': data_loader1.data, 'validation': data_loader3.data},
                 1: {'train': pd.concat([data_loader1.data_augmented, data_loader3.data_augmented, data_loader5.data_augmented], axis=0),
                     'test': data_loader2.data, 'validation': data_loader4.data},
                 2: {'train': pd.concat([data_loader1.data_augmented, data_loader4.data_augmented, data_loader5.data_augmented], axis=0),
                     'test': data_loader3.data, 'validation': data_loader2.data},
                 3: {'train': pd.concat([data_loader5.data_augmented, data_loader2.data_augmented, data_loader3.data_augmented], axis=0),
                     'test': data_loader4.data, 'validation': data_loader1.data},
                 4: {'train': pd.concat([data_loader1.data_augmented, data_loader3.data_augmented, data_loader4.data_augmented], axis=0),
                     'test': data_loader5.data, 'validation': data_loader2.data}}

    datasets3 = {0: {'train': pd.concat([data_loader3.data_augmented, data_loader3.data_augmented, data_loader5.data_augmented], axis=0),
                     'test': data_loader1.data, 'validation': data_loader4.data},
                 1: {'train': pd.concat([data_loader1.data_augmented, data_loader4.data_augmented, data_loader3.data_augmented], axis=0),
                     'test': data_loader2.data, 'validation': data_loader5.data},
                 2: {'train': pd.concat([data_loader4.data_augmented, data_loader2.data_augmented, data_loader5.data_augmented], axis=0),
                     'test': data_loader3.data, 'validation': data_loader1.data},
                 3: {'train': pd.concat([data_loader1.data_augmented, data_loader5.data_augmented, data_loader3.data_augmented], axis=0),
                     'test': data_loader4.data, 'validation': data_loader2.data},
                 4: {'train': pd.concat([data_loader2.data_augmented, data_loader1.data_augmented, data_loader4.data_augmented], axis=0),
                     'test': data_loader5.data, 'validation': data_loader3.data}}


    for k in range(5):

        train_dataset1 = MyDataset(datasets1[k]['train'])
        val_dataset1 = MyDataset(datasets1[k]['validation'])
        dicto_classes = list(train_dataset1.encode_relations.keys())


        train_dataset2 = MyDataset(datasets2[k]['train'])
        val_dataset2 = MyDataset(datasets2[k]['validation'])

        train_dataset3 = MyDataset(datasets3[k]['train'])
        val_dataset3 = MyDataset(datasets3[k]['validation'])

        if use_weights:
            weights = get_weights(pd.concat([datasets1[k]['train'], datasets1[k]['validation']], axis=0), dicto_classes).to(device).float()
        else:
            weights = None
        print('weights', weights)



        test_dataset = MyDataset(datasets1[k]['test'])

        train_dataloader1 = DataLoader(train_dataset1, batch_size=batch_size,
                                      shuffle=True, num_workers=0)
        val_dataloader1 = DataLoader(val_dataset1, batch_size=batch_size,
                                    shuffle=True, num_workers=0)

        train_dataloader2 = DataLoader(train_dataset2, batch_size=batch_size,
                                       shuffle=True, num_workers=0)
        val_dataloader2 = DataLoader(val_dataset2, batch_size=batch_size,
                                     shuffle=True, num_workers=0)

        train_dataloader3 = DataLoader(train_dataset3, batch_size=batch_size,
                                       shuffle=True, num_workers=0)
        val_dataloader3 = DataLoader(val_dataset3, batch_size=batch_size,
                                     shuffle=True, num_workers=0)

        test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=0)

        train_dataloaders = [train_dataloader1, train_dataloader2, train_dataloader2]
        val_dataloaders = [val_dataloader1, val_dataloader2, val_dataloader2]


        for i in range(len(train_dataloaders)):
            if local_model:
                model = AutoModel.from_pretrained(my_model_to_finetune)
                tokenizer = AutoTokenizer.from_pretrained(model_to_finetune)
                my_model = Model(model, tokenizer, device=device).to(device)
            elif silver_model:
                tokenizer = AutoTokenizer.from_pretrained(model_to_finetune)
                my_model = torch.load(my_model_to_finetune)
            else:
                model = AutoModel.from_pretrained(model_to_finetune)
                tokenizer = AutoTokenizer.from_pretrained(model_to_finetune)
                my_model = Model(model, tokenizer, device=device).to(device)

            finetuner = FinetuneBert(model=my_model, model_name=model_name + '_k{}_t{}'.format(k, i), alpha_contrastive=alpha_contrastive,
                                     encode_relations=train_dataset1.encode_relations, folder=folder, weights=weights)

            finetuner.train(train_dataloaders[i], val_dataloaders[i], epochs=epochs, k=k)
            finetuner.test(test_dataloader, k=k)

        if local_model:
            #model = AutoModel.from_pretrained(my_model_to_finetune)
            tokenizer = AutoTokenizer.from_pretrained(model_to_finetune)
        elif silver_model:
            tokenizer = AutoTokenizer.from_pretrained(model_to_finetune)
        else:
            #model = AutoModel.from_pretrained(model_to_finetune)
            tokenizer = AutoTokenizer.from_pretrained(model_to_finetune)

        #m1 = Model(model, tokenizer, device=device).to(device)
        m1 = torch.load('{}{}_best.pt'.format(folder, model_name + '_k{}_t{}'.format(k, 0)))
        m1.eval()

        #m2 = Model(model, tokenizer, device=device).to(device)
        m2 = torch.load('{}{}_best.pt'.format(folder, model_name + '_k{}_t{}'.format(k, 1)))
        #m2.load_state_dict(torch.load('{}{}_best.pt'.format(folder, model_name + '_k{}_t{}'.format(k, 1))))
        m2.eval()

        #m3 = Model(model, tokenizer, device=device).to(device)
        m3 = torch.load('{}{}_best.pt'.format(folder, model_name + '_k{}_t{}'.format(k, 2)))
        #m3.load_state_dict(torch.load('{}{}_best.pt'.format(folder, model_name + '_k{}_t{}'.format(k, 2))))
        m3.eval()

        ensemble = EnsembleModel(m1, m2, m3, pred_mode='combine')
        finetuner_ensemble = FinetuneBertEnsemble(ensemble, model_name=model_name + 'k_{}_ensemble'.format(k),
                                                  encode_relations=train_dataset1.encode_relations, folder=folder)
        finetuner_ensemble.test(test_dataloader, k=k)


        ensemble = EnsembleModel(m1, m2, m3, pred_mode='voting')
        finetuner_ensemble = FinetuneBertEnsemble(ensemble, model_name=model_name + 'k_{}_ensemble'.format(k),
                                                  encode_relations=train_dataset1.encode_relations, folder=folder)
        finetuner_ensemble.test(test_dataloader, k=k)

        ensemble = EnsembleModel(m1, m2, m3, pred_mode='agreement')
        finetuner_ensemble = FinetuneBertEnsemble(ensemble, model_name=model_name + 'k_{}_ensemble'.format(k),
                                                  encode_relations=train_dataset1.encode_relations, folder=folder)
        finetuner_ensemble.test(test_dataloader, k=k)


        torch.cuda.empty_cache()