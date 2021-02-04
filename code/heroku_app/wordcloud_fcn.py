import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import sys
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import words
import string
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spacy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import multiprocessing
import datetime
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')



def clean_text(url_list):
    set_of_words = set(words.words())
    stop_words = set(stopwords.words('english')+stopwords.words('spanish'))
    stop_words.add('html')
    stop_words.add('.html')
    stop_words.add('www')
    stop_words.add('http')
    stop_words.add('au')
    stop_words.add('com')
    stop_words.add('https')
    stop_words.add('htm')
    stop_words.add('php')
    stop_words.add('spip')
    stop_words.add('id')
    stop_words.add('sid')
    stop_words.add('cms')
    stop_words.add('xhtml') 
    stop_words.add('storyid')
    stop_words.add('new')
    stop_words.add('news')
    stop_words.add('whats')
    stop_words.add('people')
    stop_words.add('put')

    token = []
    for i in url_list:
        try:       
            result = re.match('.*?//.*?/(.*?)\'.*?',i)
            match = re.findall('\w\w+',result.group(1))
            token += match
        except: pass
    token = [w.lower() for w in token]
    token = [w for w in token if w in set_of_words]
    token = [w for w in token if not w in stop_words]
    return token


# return the results we use in tab_3
def return_url(df, sdg, company, date):
    '''
    df: dataset
    sdg: int
    company: string
    date: '2020-01-01'
    '''
    sample = df[(df['COMPANY']==company)&(df['date']==date)]
    urls = list(sample['SDG_'+str(sdg)+'_url'])[0][2:-1].split(', ')
    result = list(set(urls))
    if len(result)>20: # Only top 20 urls
        return result[:20]
    return result



def return_neg_score(df, company, sdg, date):
    '''
    return -10/+10 negtive scores
    '''
    result = pd.DataFrame(columns = ['date', 'score'])
    sample = df[df['COMPANY']==company]
    Date = list(sample['date'])
    score = list(sample['MA_7day_' + str(sdg)])
    ind = Date.index(date)
    start = max(ind-10,0)
    end = min(ind +11, len(Date)-1)
    for i in range(start, end):
        row={}
        if score[i] < 0:
            result = result.append({'date':Date[i],'score':score[i]}, ignore_index=True)
    del sample
    return result


def return_wordcloud(df, company, date, sdg):
    # dateList = []
    urls = []
    # for i in range(-period+1,1):
    #     dateList.append(str(datetime.datetime.strptime(date,'%Y-%m-%d') + datetime.timedelta(days=i))[:10])

    # for day in dateList:
    try:
        sample = df[(df['COMPANY']==company)&(df['date']==date)]
        url = list(sample['SDG_'+str(sdg)+'_url'])[0][2:-1].split(', ')
        urls += url
    except: pass
    
    word = clean_text(urls)
    if len(word) == 0:
        word = ['Not Available']
    cout_words = Counter(word)

    wordcloud = WordCloud(max_words =70,max_font_size=200, width=1000, height=500)
    wordcloud.generate_from_frequencies(frequencies = cout_words)
    plt.figure(figsize=(20,10) )
    plt.imshow(wordcloud, interpolation="bilinear")
    #plt.title(company + ' SDG'+str(sdg) + ' on '+ date +', score = '+str(score), {'fontsize': 18, 'color':'darkred', 'weight':'bold'})
    # codes below are used to remove white space around the saved picture
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig("word_cloud_image.png", bbox_inches = 'tight',
        pad_inches = 0)
    del sample
    del urls
    del word
    del wordcloud

    # title =  ' SDG'+str(sdg) + ' word cloud for '+ company + ' on '+ date +', in recent '+str(period)+' days'
    # return cout_words, title
