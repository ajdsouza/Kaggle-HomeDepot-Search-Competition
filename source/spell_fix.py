import requests
import re
import time
from random import randint
import pandas as pd

START_SPELL_CHECK="<span class=\"spell\">Showing results for</span>"
END_SPELL_CHECK="<br><span class=\"spell_orig\">Search instead for"

HTML_Codes = (
		("'", '&#39;'),
		('"', '&quot;'),
		('>', '&gt;'),
		('<', '&lt;'),
		('&', '&amp;'),
)

def spell_check(s):
	q = '+'.join(s.split())
	time.sleep(  randint(0,2) ) #relax and don't let google be angry
	r = requests.get("https://www.google.co.uk/search?q="+q)
	content = r.text
	start=content.find(START_SPELL_CHECK) 
	if ( start > -1 ):
		start = start + len(START_SPELL_CHECK)
		end=content.find(END_SPELL_CHECK)
		search= content[start:end]
		search = re.sub(r'<[^>]+>', '', search)
		for code in HTML_Codes:
			search = search.replace(code[1], code[0])
		search = search[1:]
	else:
		search = s
	return search ;


# read all the search strings
df_train = pd.read_csv('../kaggle-homedepot/train.csv/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('../kaggle-homedepot/test.csv/test.csv', encoding="ISO-8859-1")

# make one list of search strings
searches = list(set(pd.concat((df_train,df_test),axis=0,ignore_index=True)['search_term']))


# use google search to correct the search strings
spell_check_dict = dict()

for search in searches:
        spell_check_search = spell_check(search)
	print (search+"->" + spell_check_search)
        
        if search != spell_check_search:
                spell_check_dict[search]=spell_check_search

# save the corrected strings to to csv file
pd.DataFrame(spell_check_dict.items(), columns=['search', 'spell_corrected']).to_csv('spell_checked_search.csv',index=False)

# read the saved corrected search strings
spell_checker_df = pd.read_csv('spell_checked_search.csv', encoding="ISO-8859-1")
spell_checker = spell_checker_df.set_index('search')['spell_corrected'].to_dict()


 
