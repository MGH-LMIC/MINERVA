import scrapy


class ScraperSpider(scrapy.Spider):
    name = "scraper"

    def start_requests(self):
        urls = [
            'https://httpbin.org/ip',
            'http://ident.me/',
            'https://api.ipify.org?format=json',
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        self.logger.info('IP address: %s' % response.text)
