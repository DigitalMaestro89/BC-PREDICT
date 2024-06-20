from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# Correct the file path by using raw string or double backslashes
driver_path = 'D:\chromedriver-win64\chromedriver.exe'  # Or use 'D:\\chromedriver-win64\\chromedriver.exe'

# Initialize the WebDriver service
service = Service(driver_path)

# Initialize the WebDriver
driver = webdriver.Chrome(service=service)

try:
    # Open the desired webpage
    driver.get('https://bc.fun/game/crash')

    # Wait for the page to load
    time.sleep(5)
    # Find an element by its name attribute
    search_box = driver.find_element(By.ID, 'search2')

    # Type a query into the search box
    search_box.send_keys('Java Script')

    # Submit the search form
    search_box.send_keys(Keys.RETURN)

    # Wait for the results to load
    time.sleep(5)

    # learntocode_searchbtn BUTTON's id
    button = driver.find_element(By.ID, "learntocode_searchbtn") 
    button.click()

    # Find elements containing the search results
    results = driver.find_elements(By.CSS_SELECTOR, 'h3')

    # Extract and print the text of each result
    for result in results:
        print(result.text)

finally:
    # Close the WebDriver
    driver.quit()
