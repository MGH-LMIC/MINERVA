import json
import time
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib
from new_release_paper.normal_db.data_loader import DataLoader
from collections import defaultdict, Counter
from scipy.stats import entropy

matplotlib.rcParams.update({'font.size': 15})

def defaultlist():
    return []

class ResultsAnalyzer:
    def __init__(self, results_folder=[], data=[]):
        self.results_folder = results_folder
        data['RELATION'] = data['RELATION'].replace(['relate'], 'na')
        self.data = data

        self.classes_dict = {'positive': 0, 'negative': 1, 'na': 2, 'nan': 2}


    def get_results(self, model_names, db_name, examples_modality, temperatures, zero_shot, rephrase,
                 n_rephrasings=1, use_alternatives=False, use_COT=False, use_conf=False, results_folder=[]):

        all_results_ensemble = []
        for i in range(len(model_names)):
            model_name = model_names[i]
            if n_rephrasings > 1:
                run_name = '{}_{}_examples_{}_temps{}_zs{}_re{}_{}'.format(model_name, db_name,
                                                                           examples_modality, len(temperatures),
                                                                           int(zero_shot), int(rephrase), n_rephrasings)
            else:
                run_name = '{}_{}_examples_{}_temps{}_zs{}_re{}'.format(model_name, db_name,
                                                                        examples_modality,
                                                                        len(temperatures), int(zero_shot), int(rephrase))

            if use_alternatives:
                run_name += '_ALT'
            if use_COT:
                run_name += '_COT'
            if use_conf:
                run_name += '_CONF'


            if results_folder == []:
                results_folder = self.results_folder
            run_folder = '{}{}/'.format(results_folder[i], run_name)
            results, params = self.get_data(run_folder, run_name)
            all_results_ensemble.append(results)
        return all_results_ensemble


    def get_data(self, run_folder, run_name):
        with open('{}params.json'.format(run_folder, run_name), 'r') as f:
            params = json.load(f)

        with open('{}results.json'.format(run_folder, run_name), 'r') as f:
            results = json.load(f)

        return results, params

    def analyze_results(self, all_results, preference_order=['positive', 'negative', 'na'],
                        index=None, print_results=False, entropy_th=0):
        gt_all = []
        gt_entropy = defaultdict(defaultlist)
        gt_agreement = []
        prediction_agreement = []

        gt_agreement_entropy = []
        prediction_agreement_entropy = []
        prediction_voting = []
        data_len = min(len(v) for v in all_results.values())
        results_dict = defaultdict(defaultlist)
        entropies_dict = defaultdict(defaultlist)
        for i in range(data_len):
            row = self.data.iloc[i, :]
            relation = row['RELATION']
            row_number = 0#row['Unnamed: 0']
            bacteria = row['MICROBE']
            disease = row['DISEASE']
            sentence = row['EVIDENCE']
            last_entropies = {}

            # Get GT
            gt = self.classes_dict[relation]
            gt_all.append(gt)

            # Get predictions
            models_preds = []
            for results_name, results in all_results.items():
                voting = results[i]['self_check_scores']['voting_alternatives']

                try:
                    voting_entropy = entropy(np.array(list(voting.values())))
                except AttributeError:
                    voting_entropy = 0

                if voting_entropy <= entropy_th: # ONly results with zero entropy
                    entropies_dict[results_name].append(self.classes_dict[results[i]['prediction']])
                    gt_entropy[results_name].append(gt)
                    last_entropies[results_name] = 1

                results_dict[results_name].append(self.classes_dict[results[i]['prediction']])
                models_preds.append(results[i]['prediction'])
                
                if print_results:
                    if self.classes_dict[results[i]['prediction']] != gt_all[-1]:
                        print('{} ({})| GT: {} - Pred: {}'.format(i + 1, row_number, relation, results[i]['prediction']))
                        print('Bacteria: {} - Disease: {} | Sentence: {}'.format(bacteria, disease, sentence))
            if print_results:
                print('-'*50)

            models_preds = Counter(models_preds)

            # Get agreements
            if len(models_preds.keys()) == 1:
                gt_agreement.append(gt_all[-1])
                prediction_agreement.append(self.classes_dict[list(models_preds.keys())[0]])

                if sum(list(last_entropies.values())) == 2:
                    gt_agreement_entropy.append(gt_all[-1])
                    prediction_agreement_entropy.append(self.classes_dict[list(models_preds.keys())[0]])

            # Get voting
            selected_relation = models_preds.most_common()
            max_val = max([elem[1] for elem in selected_relation])
            selected_relation_aux = []
            for elem in selected_relation:
                if elem[1] == max_val:
                    selected_relation_aux.append(elem[0])

            if len(selected_relation_aux) > 1:
                for elem in preference_order:
                    if elem in selected_relation_aux:
                        selected_relation = elem
                        break
            else:
                selected_relation = selected_relation_aux[0]

            prediction_voting.append(self.classes_dict[selected_relation])

        # Transforming to numpy array
        gt_all = np.array(gt_all)
        gt_agreement = np.array(gt_agreement)
        prediction_agreement = np.array(prediction_agreement)
        prediction_voting = np.array(prediction_voting)
        gt_agreement_entropy = np.array(gt_agreement_entropy)
        prediction_agreement_entropy = np.array(prediction_agreement_entropy)
        acc_results = {}

        print('{} Prediction voting {}'.format('#'*25, '#'*25))
        report = classification_report(y_true=gt_all, y_pred=prediction_voting, labels=[0, 1, 2],
                                       target_names=['Positive', 'Negative', 'NA'])
        print(report)
        report = classification_report(y_true=gt_all, y_pred=prediction_voting, labels=[0, 1, 2],
                                       target_names=['Positive', 'Negative', 'NA'], output_dict=True)
        acc_results['voting_acc'] = report['accuracy']
        acc_results['voting_f1_macro'] = report['macro avg']['f1-score']
        acc_results['voting_f1_weighted'] = report['weighted avg']['f1-score']

        # Confusion matrix
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111)
        ConfusionMatrixDisplay.from_predictions(y_true=gt_all, y_pred=prediction_voting, labels=[0, 1, 2],
                                                display_labels=['Positive', 'Negative', 'NA'],
                                                xticks_rotation=30, ax=ax)
        #plt.show()

        print('{} Prediction Agreement {}'.format('#' * 25, '#' * 25))
        report = classification_report(y_true=gt_agreement, y_pred=prediction_agreement, labels=[0, 1, 2],
                                       target_names=['Positive', 'Negative', 'NA'])
        print(report)
        report = classification_report(y_true=gt_agreement, y_pred=prediction_agreement, labels=[0, 1, 2],
                                       target_names=['Positive', 'Negative', 'NA'], output_dict=True)
        print(report)
        acc_results['agreement_support'] = report['macro avg']['support']
        acc_results['agreement_acc'] = report['accuracy']
        acc_results['agreement_f1_macro'] = report['macro avg']['f1-score']
        acc_results['agreement_f1_weighted'] = report['weighted avg']['f1-score']

        fig = plt.figure(figsize=(15, 10))

        ax = fig.add_subplot(111)
        ConfusionMatrixDisplay.from_predictions(y_true=gt_agreement, y_pred=prediction_agreement,
                                                labels=[0, 1, 2],
                                                display_labels=['Positive', 'Negative', 'NA'],
                                                xticks_rotation=30, ax=ax)
        #plt.show()

        print('{} Prediction Agreement Entropy {}'.format('#' * 25, '#' * 25))
        report = classification_report(y_true=gt_agreement_entropy, y_pred=prediction_agreement_entropy, labels=[0, 1, 2],
                                       target_names=['Positive', 'Negative', 'NA'])
        print(report)
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111)
        ConfusionMatrixDisplay.from_predictions(y_true=gt_agreement_entropy, y_pred=prediction_agreement_entropy,
                                                labels=[0, 1, 2],
                                                display_labels=['Positive', 'Negative', 'NA'],
                                                xticks_rotation=30, ax=ax)
        # plt.show()

        # Independent predictions
        for results_name, result in results_dict.items():

            print('{} Prediction {} {}'.format('#' * 25, results_name, '#' * 25))
            report = classification_report(y_true=gt_all, y_pred=np.array(result), labels=[0, 1, 2],
                                           target_names=['Positive', 'Negative', 'NA'])
            print(report)
            report = classification_report(y_true=gt_all, y_pred=np.array(result), labels=[0, 1, 2],
                                           target_names=['Positive', 'Negative', 'NA'], output_dict=True)
            acc_results[results_name + '_acc'] = report['accuracy']
            acc_results[results_name + '_f1_macro'] = report['macro avg']['f1-score']
            acc_results[results_name + '_f1_weighted'] = report['weighted avg']['f1-score']

            print('{} Prediction ENTROPY {} {}'.format('#' * 25, results_name, '#' * 25))
            report = classification_report(y_true=np.array(gt_entropy[results_name]), y_pred=np.array(entropies_dict[results_name]),
                                           labels=[0, 1, 2],
                                           target_names=['Positive', 'Negative', 'NA'])

            print(report)

            report = classification_report(y_true=np.array(gt_entropy[results_name]),
                                           y_pred=np.array(entropies_dict[results_name]),
                                           labels=[0, 1, 2], output_dict=True,
                                           target_names=['Positive', 'Negative', 'NA'])
            acc_results[results_name + '_acc_entropy'] = report['accuracy']
            acc_results[results_name + '_support'] = report['macro avg']['support']
            acc_results[results_name + '_f1_macro_entropy'] = report['macro avg']['f1-score']
            acc_results[results_name + '_f1_weighted_entropy'] = report['weighted avg']['f1-score']

            # Confusion matrix
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111)
            ConfusionMatrixDisplay.from_predictions(y_true=gt_all, y_pred=np.array(result), labels=[0, 1, 2],
                                                    display_labels=['Positive', 'Negative', 'NA'],
                                                    xticks_rotation=30, ax=ax)
            #plt.show()


        return index, acc_results

if __name__ == '__main__':

    #model_names_list = [['biomistralNF', 'biomistralNF_rot1', 'biomistralNF_rot2']]
    model_names_list = [['biomistralNFLLMAUG', 'biomistralNFLLMAUG_rot1', 'biomistralNFLLMAUG_rot2']]

    all_results_models = {}
    for model_name_zero in model_names_list:
        model_name_rot0 = model_name_zero[0]
        model_name_rot1 = model_name_zero[1]
        model_name_rot2 = model_name_zero[2]

        zero_shot = True

        acc_avg = defaultdict(defaultlist)

        for k in range(5):
            print('{} Predicting K:{} {}'.format('#'*50, k + 1, '#'*50))
            db_name = 'gold_data_k{}'.format(k)
            data_path = 'initial_db/gold_data_corrected.csv'

            results_folder = ['results/{}/'.format(model_name_rot0), 'results/{}/'.format(model_name_rot1),
                              'results/{}/'.format(model_name_rot2)]

            data_loader = DataLoader(data_path=data_path, use_gold=True, split_gold=True, k=k)


            results_analyzer = ResultsAnalyzer(data=data_loader.data, results_folder=results_folder)


            # Results alternatives
            results_alternatives = results_analyzer.get_results(model_names=model_name_zero, db_name=db_name,
                                                               examples_modality='sentence',
                                                               temperatures=[1e-3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                                               zero_shot=zero_shot,
                                                               rephrase=False, n_rephrasings=1, use_COT=False, use_conf=False,
                                                               use_alternatives=True)

            all_results = {'{}'.format(model_name_rot0): results_alternatives[0], '{}'.format(model_name_rot1): results_alternatives[1],
                           '{}'.format(model_name_rot2): results_alternatives[2]}
            index, acc_results = results_analyzer.analyze_results(all_results, preference_order=['positive', 'negative', 'na'],
                                                     print_results=False)


            for key, value in acc_results.items():
                acc_avg[key].append(value)

        print(acc_avg)
        for key, value in acc_avg.items():
            result = np.array(value)
            if 'support' not in key:
                print('Result {} | Mean: {} | Std: {}'.format(key, result.mean().round(3), result.std().round(3)))
            else:
                print('{}: {} ({})'.format(key, result.sum(), result.sum()/1100*100))
