import time

import numpy as np
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from pymongo import MongoClient
import json
from metapub.pubmedcentral import get_pmcid_for_otherid, get_doi_for_otherid
from pathlib import Path
from llama_hub.file.unstructured import UnstructuredReader
import scrapy.crawler as crawler
from scrapy.utils.log import configure_logging
from multiprocessing import Process, Queue
from twisted.internet import reactor
from paper_downloader.spiders.pmc_spyder import PMCSpyder
import pymongo
import html_text
import os


class FullPapersDownloader():
    def __init__(self, mongo_db='microbiome_db', mongo_collection_source='microbiome',
                 mongo_collection_target='microbiome_full', papers_folder='all_papers/'):
        self.mongo_client = MongoClient('localhost', 2717)
        self.mongo_collection_source = self.mongo_client[mongo_db][mongo_collection_source]
        self.mongo_collection_target = self.mongo_client[mongo_db][mongo_collection_target]
        self.all_abstracts_ids = self.mongo_collection_source.distinct('_id')
        self.all_papers_ids = self.mongo_collection_target.distinct('pubmed_id')
        print(len(self.all_abstracts_ids))
        print(len(self.all_papers_ids))
        try:
            with open('lost_papers.json', 'r') as f:
                self.lost_papers = json.load(f)
        except:
            self.lost_papers = {}


        try:
            with open('not_founds.json', 'r') as f:
                self.not_founds = json.load(f)
        except:
            self.not_founds = []

        self.papers_folder = papers_folder

        # html_2_text
        self.html2text = UnstructuredReader()

        self.papers_to_download = (set(self.all_abstracts_ids) - set(self.all_papers_ids) -
                                   set(list(self.lost_papers.values())) - set(self.not_founds))
        self.papers_to_download = list(self.papers_to_download)

        # Trying to rescue the index
        self.index = 0
        self.inserted_papers = 0
        self.last_lost_papers = []

    def get_paper_text_dep(self, pcmid, folder):
        document = self.html2text.load_data(file=Path(folder + pcmid + '.html'))
        paper_text = document[0].text.lower()
        abstract_index = paper_text.find('abstract')
        references_index = paper_text.rfind('references')
        if abstract_index == -1:
            abstract_index = 0

        paper_text = paper_text[abstract_index:references_index]
        return paper_text

    def get_paper_text(self, pcmid, folder):
        HTMLFile = open("{}{}.html".format(folder, pcmid), "r").read()
        paper_text = html_text.extract_text(HTMLFile, guess_layout=False).lower()
        abstract_index = paper_text.find('abstract')
        references_index = paper_text.rfind('references')
        if abstract_index == -1:
            abstract_index = 0

        paper_text = paper_text[abstract_index:references_index]
        return paper_text

    def run_spider(self, spider, kwargs):
        def f(q):
            try:
                runner = crawler.CrawlerRunner(get_project_settings())
                deferred = runner.crawl(spider, **kwargs)
                deferred.addBoth(lambda _: reactor.stop())
                reactor.run()
                q.put(None)
            except Exception as e:
                q.put(e)

        q = Queue()
        p = Process(target=f, args=(q,))
        p.start()
        result = q.get()
        p.join()

        if result is not None:
            raise result

    def run(self):
        inserted_papers = 0
        db_length = len(self.papers_to_download)
        analyzed_papers = 0
        print('Original Length: {}'.format(db_length))

        for _id in self.papers_to_download:
        #for _id in self.all_ids[self.index:]:
            paper = self.mongo_collection_source.find_one({'_id': _id})

            pubmed_id = str(paper['_id'])
            print('Analyzing Paper: ({}|{}) {}'.format(analyzed_papers + 1, db_length, pubmed_id))
            print('Inserted full papers total: {}'.format(inserted_papers))
            publication_date = paper['publication_date']
            title = paper['title']
            journal = paper['journal']
            issn = paper['issn']
            authors = paper['authors']
            print('Paper title: {}'.format(title))
            print('Paper Journal: {}'.format(journal))
            analyzed_papers += 1

            # See if paper is in pcm
            pmcid = get_pmcid_for_otherid(pubmed_id)
            if pmcid == None:
                self.not_founds.append(pubmed_id)
                print('Paper not found')
                with open('not_founds.json', 'w') as f:
                    json.dump(self.not_founds, f)
                continue

            print('PMCID: {}'.format(pmcid))

            # Download the paper if exists
            tries = 0
            while tries < 1:
                crawler_kwargs = {'pmc_id': pmcid, 'folder': self.papers_folder}
                self.run_spider(PMCSpyder, crawler_kwargs)
                if os.path.exists("{}{}.html".format(self.papers_folder, pmcid)):
                    tries = 100
                else:
                    print('Try {}'.format(tries))
                    tries += 1
            if tries < 100:
                print('Lost Paper: {}: {} - {}'.format(analyzed_papers, pubmed_id, pmcid))
                self.lost_papers[pubmed_id] = pubmed_id
                # Saving index
                with open('lost_papers.json', 'w') as f:
                    json.dump(self.lost_papers, f)
                #self.last_lost_papers.append(analyzed_papers)
                #aux = np.array(self.last_lost_papers)
                continue

            # Once is downloaded we can retrieve it and extract the text
            paper_text = self.get_paper_text(pmcid, self.papers_folder)
            print('Inserted paper length: {}'.format(len(paper_text)))
            time.sleep(1)


            # Insert to mongo
            paper_dicto = {'title': title, 'publication_date': str(publication_date), 'full_text': paper_text,
             'journal': journal, '_id': pmcid, 'issn': issn, 'authors':authors, 'pubmed_id': pubmed_id, 'paper_num': 0}

            # Remove file
            if os.path.exists("{}{}.html".format(self.papers_folder, pmcid)):
                os.remove("{}{}.html".format(self.papers_folder, pmcid))


            try:
                self.mongo_collection_target.insert_one(paper_dicto)
                inserted_papers += 1
            except pymongo.errors.DuplicateKeyError as e:
                print(e)
                print('---')

            print('-------------------------------------------------------------------------------')







if __name__ == '__main__':
    full_papers_downloader = FullPapersDownloader(mongo_db='microbiome_abstracts_db', mongo_collection_source='microbiome_abstracts',
                 mongo_collection_target='microbiome_full_papers', papers_folder='all_papers/')
    full_papers_downloader.run()