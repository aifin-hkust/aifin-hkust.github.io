# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 15:19:41 2019
@author: ThompsonHu
"""
from selenium import webdriver
from bs4 import BeautifulSoup as bs
from datetime import datetime
import time
import re

""" Basic function to open and close chrome driver
"""
def initDriver():
    # initiate the driver
    driver = webdriver.Chrome('C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe')

    return driver

def closeDriver(driver):
    # Close the driver
	driver.close()

	return

""" Log in Twitter
"""
def loginTwitter(driver, username, password):

	# open the web page in the browser:
	driver.get('https://twitter.com/login')

	# find the boxes for username and password
	username_field = driver.find_element_by_class_name('js-username-field')
	password_field = driver.find_element_by_class_name('js-password-field')

	# enter your username:
	username_field.send_keys(username)
	driver.implicitly_wait(1)

	# enter your password:
	password_field.send_keys(password)
	driver.implicitly_wait(1)

	# click the 'Log In' button:
	driver.find_element_by_class_name('EdgeButtom--medium').click()

	return

def visitUserInfo(TwitterID):
    driver.get('https://twitter.com/' + TwitterID)
    
    # Get information in html
    strElem = driver.find_elements_by_xpath('//div[@class="ProfileNav"]/ul')
    strAll = strElem[0].text
    numAll = re.findall(r'(\w*[0-9]+)\w*', strAll)
    numTweets = numAll[0]
    numFollowing = numAll[1]
    numFollower = numAll[2]
    numLikes = numAll[3]
    print('\nTwitter ID: ' + TwitterID + '\nTweets: ' + numTweets + '\nFollowing: ' + numFollowing)
    print('Follower: ' + numFollower + '\nLikes: ' + numLikes + '\n')

    # Write information into txt file
    with open('userinfo.txt', 'w', encoding = 'gb18030') as file:
        file.write('Twitter ID: ' + TwitterID + '\r\n')
        file.write('Tweets: ' + numTweets + '\r\n')
        file.write('Following: ' + numFollowing + '\r\n')
        file.write('Follower: ' + numFollower + '\r\n')
        file.write('Likes: ' + numLikes + '\r\n')
    
    return

def GetTweets(scrollT = 6):
    for ct in range(1, scrollT):
        driver.execute_script('window.scrollBy(0,%d)' %(3000*ct))
        time.sleep(2)
    
    page_source = driver.page_source
    soup = bs(page_source,'lxml')
    tweets = []
    for li in soup.find_all('li', class_='js-stream-item'):
        # If our li doesn't have a tweet-id, we skip it as it's not going to be a tweet.
        if 'data-item-id' not in li.attrs:
            continue
        else:
            tweet = {
				'tweet_id': li['data-item-id'],
				'text': None,
				'time': None
			}
            
        # Tweet Text
        text_p = li.find('p', class_='tweet-text')
        if text_p is not None:
            tweet['text'] = text_p.get_text()
        
        # Tweet date
        date_span = li.find('span', class_='_timestamp')
        if date_span is not None:
            tweet['time'] = str(datetime.fromtimestamp(int(date_span['data-time'])))

        tweets.append(tweet)
    
    # Store the Tweets Content into txt
    f = open('TweetsContent.txt', 'w', encoding='utf-8')
    for k in range(len(tweets)):
        f.write('Time:' + tweets[k]['time'] + '\n')
        f.write('Content:' + tweets[k]['text'] + '\n')
    f.close()

if __name__ == '__main__':
    param = {}
    param['username'] = input('Input your user name:')   # twitter account
    param['password'] = input('Input your password:')   # password
    param['TwitterID'] = input('Input Twitter ID:')   # Twitter ID
    driver = initDriver()
    
    loginTwitter(driver, param['username'], param['password'])
    time.sleep(3)
    
    visitUserInfo(param['TwitterID'])
    
    GetTweets()

    closeDriver(driver)