import time
import csv
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from urllib.parse import urljoin

BASE_URL = "https://www.shl.com"

def get_duration_from_detail_page(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        duration_tag = soup.find("p", string=lambda x: x and "Approximate Completion Time" in x)
        if duration_tag:
            return duration_tag.text.strip().split("=")[-1].strip()
    except Exception as e:
        print(f"Error fetching duration from {url}: {e}")
    return "N/A"

# Set up Selenium
options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)
driver.get("https://www.shl.com/products/product-catalog/")
time.sleep(5)  # Let page load

# Revert to original working row selector
rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
print(f"Found {len(rows)} assessments.")

data = []

for row in rows:
    try:
        cols = row.find_elements(By.TAG_NAME, "td")
        if not cols or len(cols) < 4:
            continue

        # Assessment name and detail URL
        link_elem = cols[0].find_element(By.TAG_NAME, "a")
        name = link_elem.text.strip()
        detail_url = urljoin(BASE_URL, link_elem.get_attribute("href"))

        # Remote and Adaptive support
        remote = "Yes" if "catalogue__circle -yes" in cols[1].get_attribute("innerHTML") else "No"
        adaptive = "Yes" if "catalogue__circle -yes" in cols[2].get_attribute("innerHTML") else "No"

        # Test types
        type_spans = cols[3].find_elements(By.CLASS_NAME, "product-catalogue__key")
        test_types = ", ".join([span.text.strip() for span in type_spans]) if type_spans else "N/A"

        # Duration from detail page
        duration = get_duration_from_detail_page(detail_url)

        data.append({
            "Name": name,
            "URL": detail_url,
            "Duration (mins)": duration,
            "Remote Testing Support": remote,
            "Adaptive/IRT Support": adaptive,
            "Test Types": test_types
        })

    except Exception as e:
        print(f"Error processing row: {e}")
        continue

driver.quit()

# Save to CSV
csv_filename = "shl_product_catalog.csv"
with open(csv_filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)

print(f"✅ Done! {len(data)} assessments saved to '{csv_filename}'")
