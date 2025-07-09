from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
import concurrent.futures

def clean_text(text):
    clean_text = re.sub(r'Request an appointment|Learn more about.*?|Subscribe|Address 1.*', '', text, flags=re.DOTALL)
    clean_text = re.sub(r'[\n\r\t]+', ' ', clean_text)  # Remove excessive newlines/tabs
    clean_text = re.sub(r' {2,}', ' ', clean_text)  # Replace multiple spaces with one
    return clean_text.strip()

def scrape_disease_info(disease_name):
    disease_info = {}
    try:
        # Set up Selenium with Chromium (or Chrome)
        chrome_options = Options()
        # chrome_options.add_argument("--headless")  # Run in headless mode (no GUI)
        # chrome_options.add_argument("--no-sandbox")
        # chrome_options.add_argument("--disable-dev-shm-usage")
        # chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration
        # chrome_options.add_argument("--window-size=1920,1080")  # Set a specific window size

        driver_path = r"C:\Users\ayush\Desktop\chromedriver.exe"
        service = Service(driver_path)
        driver = webdriver.Chrome(service=service, options=chrome_options)

        # Step 1: Open Mayo Clinic search page
        driver.get("https://www.mayoclinic.org/")

        # Step 2: Locate the search bar and input disease name
        search_bar = WebDriverWait(driver, 2).until(
            EC.presence_of_element_located((By.ID, 'search-input-globalsearch-a02e2c35b8'))
        )
        search_bar.send_keys(disease_name)
        search_bar.send_keys(Keys.RETURN)

        # Step 3: Click on the first result link
        first_result = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH,'//*[@id="cmp-skip-to-main__content"]/ul/li[1]/h3/div/a'))
        )
        first_result.click()

        # Step 4: Scrape the disease information
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Extract key information: Symptoms, Causes, Treatment
        title = soup.find('h1').get_text(strip=True)
        disease_info['title'] = title

        headings = soup.find_all('h2')

        for heading in headings:
            section_title = heading.get_text(strip=True).lower()
            section_content = ""

            next_sibling = heading.find_next_sibling()
            while next_sibling and next_sibling.name != 'h2':
                section_content += clean_text(next_sibling.get_text(strip=True)) + "\n"
                next_sibling = next_sibling.find_next_sibling()

            if "symptoms" in section_title:
                disease_info['symptoms'] = section_content.strip()
            elif "causes" in section_title:
                disease_info['causes'] = section_content.strip()
            elif "treatment" in section_title:
                disease_info['treatment'] = section_content.strip()

        # Provide default information if sections are not found
        disease_info['symptoms'] = disease_info.get('symptoms', "No symptoms information available.")
        disease_info['causes'] = disease_info.get('causes', "No causes information available.")
        disease_info['treatment'] = disease_info.get('treatment', "No treatment information available.")

    except Exception as e:
        print(f"An error occurred: {e}")
        disease_info['title'] = "disease not found"
        disease_info['symptoms'] = "No symptoms information available."
        disease_info['causes'] = "No causes information available."
        disease_info['treatment'] = "No treatment information available."
    finally:
        driver.quit()

    return disease_info

# Usage of concurrent.futures for parallel scraping
def scrape_multiple_diseases(disease_list):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(scrape_disease_info, disease_list))
    return results

# Example usage
# disease_list = ["kidney stones", "diabetes", "asthma"]
# results = scrape_multiple_diseases(disease_list)
# for info in results:
#     print(f"Title: {info['title']}")
#     print(f"Symptoms: {info['symptoms']}")
#     print(f"Causes: {info['causes']}")
#     print(f"Treatment: {info['treatment']}")
