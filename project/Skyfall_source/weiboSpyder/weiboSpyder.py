from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import re

# 全局变量
driver = webdriver.Chrome("C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe")

def loginWeibo(username, password):
    driver.get('https://www.weibo.com/hk')
    time.sleep(10)

    driver.find_element_by_id("loginname").send_keys(username)
    time.sleep(3)
    driver.find_element_by_name("password").send_keys(password)
    time.sleep(3)
    driver.find_element_by_name("password").send_keys(Keys.ENTER)

def visitUserInfo(userId):
    driver.get('https://weibo.com/' + userId + '?profile_ftype=1&is_all=1')
    # 用户id
    print('用户id:' + userId)
    # 用户昵称
    strName = driver.find_element_by_xpath("//div[@class='pf_username']/h1")
    name = strName.text
    print('昵称:' + name)
    # 微博数、粉丝数、关注数
    strElem = driver.find_elements_by_xpath("//table[@class='tb_counter']/tbody/tr/td")
    strGz = strElem[0].text    # 关注数  
    numGz = re.findall(r'(\w*[0-9]+)\w*', strGz)  
    strFs = strElem[1].text    # 粉丝数  
    numFs = re.findall(r'(\w*[0-9]+)\w*', strFs)  
    strWb = strElem[2].text    # 微博数  
    numWb = re.findall(r'(\w*[0-9]+)\w*', strWb) 
    print("关注数：" + numGz[0])
    print("粉丝数：" + numFs[0])
    print("微博数：" + numWb[0])
    print('\n')
    
    # 将用户信息写到文件里
    with open("userinfo.txt", "w", encoding = "gb18030") as file:
        file.write("用户ID：" + userId + '\r\n')
        file.write("昵称：" + name + '\r\n')
        file.write("微博数：" + numWb[0] + '\r\n')
        file.write("关注数：" + numGz[0] + '\r\n')
        file.write("粉丝数：" + numFs[0] + '\r\n')
        
def visitWeiboContent(userId):
    url = 'https://weibo.com/u/' + userId + '?is_ori=1'
    # 获取微博内容及发布时间
    flag1 = 1
    ct = 1
    while flag1:
        try:
            driver.find_element_by_xpath('//a[@class="page next S_txt1 S_line1"]')
            break
        except:
            driver.execute_script('window.scrollBy(0,%d)' %(3000*ct))
            time.sleep(5)
            ct += 1
    driver.execute_script("document.getElementsByClassName('layer_menu_list W_scroll')[0].style.display='block'")
    page = driver.find_element_by_xpath('//div[@class="layer_menu_list W_scroll"]/ul/li[1]/a').text
    page_num = re.findall('\d+',page)
    pretime = []
    content = []
    for n in range(0,int(page_num[0])+1):
        driver.get(url+'&page=%d' %(n+1))
        flag1 = 1
        ct = 1
        while flag1:
            try:
                driver.find_element_by_xpath('//div[@class="W_pages"]/a[@class="page next S_txt1 S_line1"]')
                break
            except:
                driver.execute_script('window.scrollBy(0,%d)' %(3000*ct))
                time.sleep(3)
                ct += 1
                if ct == 5: # 到了尾页，跳出循环，经验判断，一页不会下拉到5次之多
                    break
                
        iter1 = driver.find_elements_by_xpath('//div[@class="WB_detail"]')
        for i in range(0,len(iter1)):
            pretime1 = iter1[i].find_element_by_xpath('div[@class="WB_from S_txt2"]/a[1]').text
            pretime.append(pretime1)
            content1 = iter1[i].find_element_by_xpath('div[@class="WB_text W_f14"]').text
            content.append(content1)
            print('Finished loading ' + str(i+1) + ' nd content!')
            
    #存储数据在文本中
    f = open('weiboContent.txt', "w",encoding='utf-8')
    for k in range(len(content)):
        f.write("时间:"+pretime[k]+'\n')
        f.write("内容:"+content[k]+'\n')
    f.close()
        
if __name__ == '__main__':
    param = {}
    param['username'] = input('请输入账号:')   # 输入微博账号
    param['password'] = input('请输入密码:')   # 输入密码
    param['uid'] = input('微博用户ID:')        # uid = '5477684004'
    loginWeibo(param['username'], param['password'])
    time.sleep(10)
    
    visitUserInfo(param['uid'])                        # 获取用户基本信息
    visitWeiboContent(param['uid'])                    # 获取微博内容