import os
import json
import requests
import re
from typing import List, Dict

class ZendeskAPIScraper:
    def __init__(self):
        self.base_url = "https://joinvoy.zendesk.com/api/v2/help_center/en-gb"
        self.faq_data = []

    def get_all_articles(self) -> List[Dict]:
        """Get all articles using Zendesk API."""
        articles = []
        url = f"{self.base_url}/articles.json"
        
        while url:
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                break
                
            data = response.json()
            articles.extend(data['articles'])
            
            # Check if there are more pages
            url = data['next_page']
            print(f"Fetched {len(data['articles'])} articles. Next page: {url}")
        
        return articles

    def process_articles(self, articles: List[Dict]) -> List[Dict[str, str]]:
        """Process articles from API response."""
        for article in articles:
            try:
                # Get full article content
                article_url = f"{self.base_url}/articles/{article['id']}.json"
                response = requests.get(article_url)
                if response.status_code != 200:
                    print(f"Error getting article {article['id']}: {response.status_code}")
                    continue
                
                full_article = response.json()['article']
                # remove all html tags
                full_article['body'] = re.sub(r'<[^>]*>', '', full_article['body'])
                self.faq_data.append({
                    'url': full_article['html_url'],
                    'title': full_article['title'],
                    'content': full_article['body']
                })
                print(f"Processed: {full_article['title']}")
            except Exception as e:
                print(f"Error processing article {article['id']}: {str(e)}")
        
        return self.faq_data

    def scrape_all_faqs(self) -> List[Dict[str, str]]:
        """Scrape all FAQ articles using Zendesk API."""
        articles = self.get_all_articles()
        print(f"Found {len(articles)} articles via API")
        return self.process_articles(articles)

    def save_to_json(self, output_path: str = '../data/faq_data.json'):
        """Save scraped data to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.faq_data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(self.faq_data)} articles to {output_path}")

if __name__ == "__main__":
    scraper = ZendeskAPIScraper()
    faq_data = scraper.scrape_all_faqs()
    scraper.save_to_json()
    print(f"Scraped {len(faq_data)} articles successfully.")