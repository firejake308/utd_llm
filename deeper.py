from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import TimeoutException
from csv import DictReader, DictWriter

options = webdriver.ChromeOptions()
# options.add_experimental_option("detach", True)
driver = webdriver.Chrome()
final_pages = []
try:
    with open('articles.csv') as f:
        reader = DictReader(f)
        for row in reader:
            driver.get(row['href'])
            try:
                els = WebDriverWait(driver, timeout=5).until(lambda d: d.find_elements(By.CSS_SELECTOR, value=".toc-result-item a"))
                print(f"Found {len(els)} articles within {row['title']}")

                for el in els:
                    final_pages.append({'title': el.get_attribute('textContent'), 'href': el.get_attribute('href')})
            except TimeoutException:
                final_pages.append(row)
                print(f"Unable to find deeper articles for {row['title']}")
finally:
    driver.quit()
    print(f"{len(final_pages)} final pages found")

    with open('final_articles.csv', mode='w', encoding='utf-8') as f:
        writer = DictWriter(f, ['title', 'href'])
        writer.writeheader()
        writer.writerows(final_pages)