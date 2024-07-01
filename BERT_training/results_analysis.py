import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os

class ResultsAnalyzer:
    def __init__(self, model_name, results_folder='results/'):
        self.model_name = model_name
        self.results_folder = results_folder + '{}/'.format(model_name)

    def show_results_independet(self, t=0):
        avg_accuracy = []
        avg_f1 = []
        results = {'k{}'.format(k): self.results_folder + '{}_k{}_t{}_report.pkl'.format(model_name, k, t) for k in range(5)}
        results = {k: torch.load(v) for k, v in results.items()}
        for k, v in results.items():
            print('{}: Accuracy {}'.format(k, v['accuracy']))
            avg_accuracy.append(v['accuracy'])
            avg_f1.append(v['weighted avg']['f1-score'])

        print('Avg Acc: {} | Avg F1: {}'.format(np.array(avg_accuracy).mean(), np.array(avg_f1).mean()))

    def show_results_ensemble(self, type='combine'):
        avg_accuracy = []
        results_combined = {'k{}'.format(k): self.results_folder + '{}k_{}_ensemble_report_{}.pkl'.format(model_name, k, type) for k in range(5)}
        results_combined = {k: torch.load(v) for k, v in results_combined.items()}
        total_support = []
        n = 220*5
        for k, v in results_combined.items():
            print('{}: Accuracy {} (Support: {})'.format(k, v['accuracy'], v['macro avg']['support']))
            total_support.append(v['macro avg']['support'])
            avg_accuracy.append(v['accuracy'])
        print('Avg: {}'.format(np.array(avg_accuracy).mean()))
        print('Avg support: {}'.format(np.array(total_support).sum()/n))

if __name__ == '__main__':

    model_name = 'Bio_ClinicalBERT'

    results_folder = 'results_3labels/'
    result_analyzer = ResultsAnalyzer(model_name, results_folder=results_folder)

    print('Results t=0')
    result_analyzer.show_results_independet(t=0)
    print('')

    print('Results t=1')
    result_analyzer.show_results_independet(t=1)
    print('')

    print('Results t=2')
    result_analyzer.show_results_independet(t=2)
    print('')

    print('Results combine')
    result_analyzer.show_results_ensemble(type='combine')
    print('')

    print('Results voting')
    result_analyzer.show_results_ensemble(type='voting')
    print('')

    print('Results agreement')
    result_analyzer.show_results_ensemble(type='agreement')
    print('')
