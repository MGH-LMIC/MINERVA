import pandas as pd

class DbStatistics:
    def __init__(self, paths=[]):
        self.paths = paths
        self.dbs = [pd.read_csv(elem) for elem in paths]

    def get_statistics(self):
        for i in range(len(self.dbs)):
            name = self.paths[i]
            bacterias = self.dbs[i]['b_cui']
            diseases = self.dbs[i]['d_cui']
            relations = self.dbs[i]['relation']

            print('Statistics for {}:'.format(name))
            bacterias = list(set(bacterias.values.tolist()))
            diseases = list(set(diseases.values.tolist()))
            print('Number of bacterias: {}'.format(len(bacterias)))
            print('Number of diseases: {}'.format(len(diseases)))
            print('Number of relations: {}'.format(len(relations)))





if __name__ == '__main__':
    paths = ['AMADIS/AMADIS_new.csv', 'GMDAD/GMDAD_new.csv', 'HMDAD/HMDAD_new.csv', 'Original/Original_new.csv']

    statister = DbStatistics(paths=paths)
    statister.get_statistics()