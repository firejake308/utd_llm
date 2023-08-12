import torch.nn.functional as F
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import logging

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support.expected_conditions import invisibility_of_element_located
<<<<<<< HEAD
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
=======
from selenium.common.exceptions import TimeoutException
>>>>>>> d089900a4d001f6c001b81c5a218445ef814ddf8
from csv import DictReader, DictWriter
import re
import dotenv
import os
import threading
import pinecone

dotenv.load_dotenv()

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

tokenizer = None
model = None
def load_model():
    global tokenizer
    global model
    model_name = 'intfloat/e5-large-v2'
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(('Done loading tokenizer'))
    print('Loading model...')
    model = AutoModel.from_pretrained(model_name)
    print('Done loading model')
model_thread = threading.Thread(target=load_model)
model_thread.start()

pinecone.init(api_key=os.getenv('PINEKEY'), environment="us-west4-gcp-free")
index = pinecone.Index("utd-llm-para")

def init_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument("window-size=1400,1500")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("start-maximized")
    options.add_argument("enable-automation")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-dev-shm-usage")
    options.add_experimental_option("detach", True)
    driver = webdriver.Chrome(options)

    # shoot, i'm gonna have to log in if i actually want to read the articles
    driver.get('https://uptodate.com/login')
    un_inp = WebDriverWait(driver, timeout=5).until(lambda d: d.find_element(By.ID, value="userName"))
    un_inp.send_keys(os.getenv('UN'))
    pw_inp = driver.find_element(By.ID, value="password")
    pw_inp.send_keys(os.getenv('PW'))
    pw_inp = driver.find_element(By.ID, value="password")
    login_btn = driver.find_element(By.CLASS_NAME, value="login-button")
    login_btn.click()
    driver.implicitly_wait(5)
    print('Logged in')
    return driver
driver = init_driver()
skipped_articles = []
try:
    articles = pd.read_csv('final_articles.csv').drop_duplicates('href').sample(frac=1, random_state=4242)
    article_num = -1
    for row in articles.itertuples():
        # to help with skipping later
        article_num += 1
        print('Fetching article', article_num)
        if article_num not in [1658, 2331, 4992]:
            continue
        url_stub = row.href[len('https://www.uptodate.com/contents/'):]
        if url_stub.startswith('society-guideline-links') or url_stub.startswith('table-of-contents'):
            print('Skipping', row.href)
            continue
        driver.get(row.href)
        
        try:
            topic_text = WebDriverWait(driver, timeout=10).until(lambda d: d.find_element(By.ID, value="topicText"))
            WebDriverWait(driver, timeout=5).until(invisibility_of_element_located((By.CSS_SELECTOR, 'a[href="/store"]')))
        except TimeoutException:
            print(f"Timed out on #{article_num}: {url_stub}")
            # probably some sort of rate-limiting, so eventually, we'll probably sleep it off
            # but for now, we'll try printing the HTML once
            # print(driver.find_element(By.TAG_NAME, "body").get_attribute("innerText"))
            # looked like the search page for some reason. Imma try turning it off and on
            driver.quit()
            driver = init_driver()
            driver.get(row.href)
            try:
                topic_text = WebDriverWait(driver, timeout=10).until(lambda d: d.find_element(By.ID, value="topicText"))
                WebDriverWait(driver, timeout=5).until(invisibility_of_element_located((By.CSS_SELECTOR, 'a[href="/store"]')))
            except:
                print('Double-timeout on {article_num}, skipping')
                skipped_articles.append(article_num)
                continue
        try:
            topic_html = topic_text.get_attribute('innerText')
        except StaleElementReferenceException:
            topic_text = WebDriverWait(driver, timeout=5).until(lambda d: d.find_element(By.ID, value="topicText"))
            topic_html = topic_text.get_attribute('innerText')
        paras = re.split(r'\n+', topic_html)
        print(f"Found {len(paras)} paragraphs within {row.title}")
    
        para_count = 0
        curr_header = None
        titles = []
        texts = []
        curr_text = 'passage: '
        for para in paras:
            if para.isupper():
                # if curr_header is None, then this is just INTRODUCTION and nothing's been said yet
                if curr_header is not None:
                    titles.append(curr_header)
                    texts.append(curr_text)
                # but we still need to reset and prepare the next section
                curr_header = row.href[len('https://www.uptodate.com/contents/'):] + '/#' + para.strip().encode('ascii', errors='xmlcharrefreplace').decode()
                curr_text = 'passage: '
            elif len(para) > 3:
                curr_text += para + '\n'
        if curr_header is None:
            curr_header = 'EVERYTHING'
        titles.append(curr_header)
        texts.append(curr_text)
        print(f"Found {len(texts)} sections")
        for title in titles:
            print(title)

        # Tokenize the input texts
        if model is None:
            model_thread.join()
        batch_dict = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        print("Tokenized batch_dict")

        with torch.no_grad():
            attn_mask = batch_dict['attention_mask']
            outputs = model(input_ids=batch_dict['input_ids'],
                            token_type_ids=batch_dict['token_type_ids'],
                            attention_mask=attn_mask)
            embeddings = average_pool(outputs.last_hidden_state,attn_mask)
            index.upsert(list(zip(titles, embeddings.tolist())))
            print('Updated index')
except Exception as e:
    logging.exception(e)
finally:
    driver.quit()
    print(skipped_articles)
    print('Done')
