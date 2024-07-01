import pandas as pd

class DulpicatesSolver:
    def __init__(self):
        pass

    def run(self, db_path):
        print('Analyzing: {}'.format(db_path))

        db_name = db_path.split('/')[0]
        db_folder = db_name

        db = pd.read_csv(db_path)
        db['rels'] = db['b_cui'] + '_' + db['d_cui']
        groups = db.groupby(by='rels')
        edited_groups = []
        for g_name, group in groups:
            if group.shape[0] > 1:
                relations = group['relation'].value_counts().sort_values(ascending=False)
                final_rel = relations.index[0]
                edited_group = group.iloc[-1, :]
                del edited_group['rels']
                edited_group['relation'] = final_rel
                edited_groups.append(edited_group.to_frame('hola').transpose())
            else:
                del group['rels']
                edited_groups.append(group)
        edited_groups = pd.concat(edited_groups, axis=0)

        print('Saving: {}'.format(db_name))
        edited_groups.to_csv('{}/{}_newNoRepeated.csv'.format(db_folder, db_name))






if __name__ == '__main__':

    db_paths = ['AMADIS/AMADIS_new.csv', 'GMDAD/GMDAD_new.csv', 'HMDAD/HMDAD_new.csv', 'Original/Original_new.csv',
                'Disbiome/Disbiome_new.csv']


    for db_path in db_paths:
        solver = DulpicatesSolver()
        solver.run(db_path)

