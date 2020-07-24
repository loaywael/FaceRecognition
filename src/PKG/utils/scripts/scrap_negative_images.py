import io
import pandas
import hashlib
import requests
import threading
from PIL import Image
from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager


def get_content(url):
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get(url)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    page_content = driver.page_source
    driver.quit()
    return page_content


def parse_image_urls(content, classes, location, source):
    soup = BeautifulSoup(content)
    results = []
    for a in soup.findAll(attr={"class": classes}):
        name = a.find(location)
        if name not in results:
            results.append(name.get(source))
    return results


def save_urls_to_csv(image_urls):
    df = pd.DataFrame({"links": image_urls})
    df.to_csv("links.csv", index=False, encoding="utf-8")


def save_image(image_url, output_dir):
    response = requests.get(image_url, headers={"User-agent": "Mozilla/5.0"})
    image_content = response.content
    image_file = io.BytesIO(image_content)
    image = Image.open(image_file).convert("RGB")
    file_name = hashlib.sha1(image_content).hexdigest()[:10] + ".png"
    image.save(output_dir/file_name, "PNG", quality=80)


if __name__ == "__main__":
    url = "https://duckduckgo.com/?q=dogs&t=brave&iax=images&ia=images"
    content = get_content(url)
    image_urls = parse_image_urls(content, "any", "img", "src")
    save_urls_to_csv(image_urls, "../data/negative/")
