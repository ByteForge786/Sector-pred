import requests
from bs4 import BeautifulSoup

def get_company_description(company_name):
    search_url = f"https://en.wikipedia.org/w/index.php?search={company_name}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    search_results = soup.find_all('div', class_='mw-search-result-heading')
    
    if not search_results:
        return "Company description not found on Wikipedia."
    
    first_result = search_results[0]
    link = first_result.find('a')['href']
    full_url = f"https://en.wikipedia.org{link}"
    
    response = requests.get(full_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    description = soup.find('p').text.strip()
    
    return description

company_name = "Apple Inc."
description = get_company_description(company_name)
print(description[:200])  # Print first 200 characters of the description
