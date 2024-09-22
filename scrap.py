import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Starting URL
url = 'https://www.indianyellowpages.com/abohar/basant-nagar-local-business-directory.htm'

# Request the page
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Extract all links and join with base URL if relative
links = [urljoin(url, link.get('href')) for link in soup.find_all('a', href=True)]

# Filter subdomain links
subdomain_links = [link for link in links if 'subdomain.example.com' in link]

# Print or process the subdomain links
for link in subdomain_links:
    print(link)
