import metapub.exceptions
import pandas as pd
from metapub import PubMedFetcher
from pymongo import MongoClient
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pymongo
import time
import metapub
import json
import eutils

class MyPubMedReader:
    def __init__(self, mongo_collection='microbiome'):
        self.mongo_client = MongoClient('localhost', 2717)
        self.mongo_collection = self.mongo_client['{}_db'.format(mongo_collection)][mongo_collection]
        self.pubmed = PubMedFetcher()#PubMed(tool="MyTool", email="my@email.address")

        try:
            with open('pubmed_reader_dates.json', 'r') as f:
                self.dates_ready = json.load(f)
        except:
            self.dates_ready = []

    def insert_articles(self, from_year=2015, to_year=2024):
        # Dicto results
        results_dict = {}

        # Iterating monthly to get everything
        from_date = '{}/01/01'.format(from_year)
        from_date_object = datetime.strptime(from_date, '%Y/%m/%d').date()
        to_date = '{}/03/31'.format(to_year)
        to_date_object = datetime.strptime(to_date, '%Y/%m/%d').date()
        months_difference = (to_date_object.year - from_date_object.year)*12 + to_date_object.month - from_date_object.month + 1

        # Making the monthly queries
        for i in range(months_difference):
            to_date_object = from_date_object + relativedelta(months=1)
            query = (('(((((microbiome) OR (microbiota)) OR (dysbiosis)) OR (microbiome alterations)) '
                     'AND (("{}"[Date - Create] : "{}"[Date - Create]))) AND (English[Language])')
                     .format(from_date_object.strftime(format='%Y/%m/%d'), to_date_object.strftime(format='%Y/%m/%d')))

            results = self.pubmed.pmids_for_query(query, retstart=0, retmax=9999)
            n_results = 0
            print('From: {} - To: {} | N articles: {}'.format(from_date_object, to_date_object, len(results)))
            self.dates_ready.append('From: {} - To: {} | N articles: {}'.format(from_date_object, to_date_object, len(results)))
            for pubmed_id in results:
                try:
                    article = self.pubmed.article_by_pmid(pubmed_id)
                except metapub.exceptions.MetaPubError as e:
                    print(e)
                    continue
                except eutils._internal.exceptions.EutilsNCBIError as e:
                    print(e)
                    continue

                try:
                    title = article.title.strip()
                except:
                    continue
                try:
                    publication_date = article.year
                except:
                    continue

                try:
                    abstract = article.abstract.strip()
                    if abstract == None:
                        continue
                except:
                    continue

                try:
                    journal = article.journal
                except:
                    journal = None

                try:
                    authors = article.authors
                except:
                    authors = None

                try:
                    doi = article.doi
                except:
                    doi = None

                try:
                    issn = article.issn
                except:
                    issn = None

                results_dict[pubmed_id] = {'title': title, 'publication_date': str(publication_date), 'abstract': abstract,
                                           'journal': journal, 'doi': doi, 'authors': authors, 'issn': issn,
                                           '_id': pubmed_id}
                print('{}) date: {} | title: {}'.format(n_results + 1, publication_date, title))
                try:
                    self.mongo_collection.insert_one(results_dict[pubmed_id])
                except pymongo.errors.DuplicateKeyError as e:
                    print(e)
                    print(results_dict[pubmed_id]['title'])
                    print(results_dict[pubmed_id]['_id'])
                    print('---')
                time.sleep(0.3)
                n_results += 1

            from_date_object = to_date_object

            with open('pubmed_reader_dates.json', 'w') as f:
                json.dump(self.dates_ready, f)








if __name__ == '__main__':
    pubmed_reader = MyPubMedReader(mongo_collection='microbiome_abstracts')
    pubmed_reader.insert_articles(from_year=2013, to_year=2024)