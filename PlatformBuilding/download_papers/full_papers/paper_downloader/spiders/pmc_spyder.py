from pathlib import Path
import scrapy
import random

class PMCSpyder(scrapy.Spider):
    name = "pmc_spyder"


    def __init__(self, pmc_id='', folder='all_papers/'):
        self.pmc_id = pmc_id
        self.folder = folder


    def start_requests(self, ):
        urls = ["https://www.ncbi.nlm.nih.gov/pmc/articles/{}/".format(self.pmc_id)]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = "{}{}.html".format(self.folder, self.pmc_id)
        Path(filename).write_bytes(response.body)
        self.log(f"Saved file {filename}")
