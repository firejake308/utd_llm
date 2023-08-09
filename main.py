from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from csv import DictWriter

driver = webdriver.Chrome()
pages = []
ignore = ["Practice Changing UpDates", "What's New", "Patient Education", "Calculators", "Authors and Editors"]
articles = []
try:
    driver.get('https://www.uptodate.com/contents/table-of-contents')
    els = WebDriverWait(driver, timeout=3).until(lambda d: d.find_elements(By.CSS_SELECTOR, value=".toc-result-item a"))
    for el in els:
        if el.text not in ignore:
            pages.append(el.get_attribute('href'))
    print(f"Found {len(pages)} pages")

    for page in pages:
        driver.get(page)
        
        els = WebDriverWait(driver, timeout=10).until(lambda d: d.find_elements(By.CSS_SELECTOR, value=".toc-result-item a"))
        print(f"Found {len(els)} articles on this page [{page}]")
        for el in els:
            articles.append({'title': el.text, 'href': el.get_attribute('href')})
finally:
    driver.quit()

    with open('articles.csv', 'w') as f:
        writer = DictWriter(f, ['title', 'href'])
        writer.writeheader()
        writer.writerows(articles)