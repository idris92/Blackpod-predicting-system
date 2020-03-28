import os
import time
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
#from selenium.webdriver.support.select.Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import StaleElementReferenceException



chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')


def chrome_drive():
	driver = webdriver.Chrome(executable_path = '/home/devtotti/Workspace/extensions/chromedriver_linux64/chromedriver')#, options=chrome_options)

	driver.wait = WebDriverWait(driver, 5)

	return driver

def get_data(driver):
	driver.get("https://www.accuweather.com/en/ng/ife/255020/march-weather/255020?year=2020&view=list")
	#driver.get("/home/devtotti/Workspace/blackpod//home/devtotti/Workspace/blackpod/Ife, Osun, Nigeria Monthly Weather | AccuWeather.html")
	#driver.wait(5)

	time.sleep(5)
	temps = driver.find_elements_by_class_name("temps")

	for temp in temps:
		temperature = temp.text
		temperature.split(" / ")
		print(temperature[0],temperature[1])








driver = chrome_drive()
get_data(driver)

