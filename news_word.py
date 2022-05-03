# 감성 분류 모델 구축
# 깃허브 데이터 파일 참조 : https://github.com/e9t/nsmc
import pandas as pd

nsmc_train_df = pd.read_csv('data/ratings_train.txt', 
                            encoding='utf8', sep='\t', engine='python')
nsmc_train_df.head()

nsmc_train_df.info()

nsmc_train_df = nsmc_train_df[nsmc_train_df['document'].notnull()]

nsmc_train_df.info()

nsmc_train_df['label'].value_counts()

import re

nsmc_train_df['document'] =\
 nsmc_train_df['document'].apply(
     lambda x : re.sub(r'[^ ㄱ-ㅣ가-힣]+', " ", x))
 
nsmc_train_df.head()

#%%

nsmc_test_df = pd.read_csv('data/ratings_test.txt', 
                           encoding='utf8', sep='\t', engine='python')
nsmc_test_df.head()

nsmc_test_df.info()

nsmc_test_df = nsmc_test_df[nsmc_test_df['document'].notnull()]


print(nsmc_test_df['label'].value_counts())

nsmc_test_df['document'] = \
nsmc_test_df['document'].apply(
    lambda x : re.sub(r'[^ ㄱ-ㅣ가-힣]+', "", x))

#%%
from konlpy.tag import Okt
okt = Okt()

def okt_tokenizer(text) :
    tokens = okt.morphs(text)
    return tokens

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(tokenizer=okt_tokenizer, ngram_range=(1,2),
                        min_df=3, max_df=0.9)
tfidf.fit(nsmc_train_df['document'])
nsmc_train_tfidf = tfidf.transform(nsmc_train_df['document'])

#%%
from sklearn.linear_model import LogisticRegression

SA_lr = LogisticRegression(random_state=7)
SA_lr.fit(nsmc_train_tfidf, nsmc_train_df['label'])

from sklearn.model_selection import GridSearchCV

params = {'C' : [1, 3, 3.5, 4, 4.5, 5]}
SA_lr_grid_gv = GridSearchCV(SA_lr, param_grid=params, cv=3,
                             scoring='accuracy', verbose=1)

SA_lr_grid_gv.fit(nsmc_train_tfidf, nsmc_train_df['label'])
print(SA_lr_grid_gv.best_params_, round(SA_lr_grid_gv.best_score_, 4))

SA_lr_best = SA_lr_grid_gv.best_estimator_

nsmc_test_tfidf = tfidf.transform(nsmc_test_df['document'])
test_predict = SA_lr_best.predict(nsmc_test_tfidf)

from sklearn.metrics import accuracy_score

print('감성 분석 정확도 : ', 
      round(accuracy_score(nsmc_test_df['label'], test_predict), 3))
#%%
st = input('감성 분석할 문장 입력 >> ')

st = re.compile(r'[ㄱ-ㅣ가-힣]+').findall(st)
print(st)
st = [" ".join(st)]
print(st)

st_tfidf = tfidf.transform(st)

st_predict = SA_lr_best.predict(st_tfidf)

if (st_predict == 0) :
    print(st, "->> 부정 감성")
else :
    print(st, "->> 긍정 감성") 
    
#%%
import json

file_name = '축구_naver_news'
with open(file_name+'.json', 
          encoding = 'utf8') as j_f:
    data = json.load(j_f)
type(data)
print(data)

data_title = []
data_description = []

for item in data:
 data_title.append(item['title'])
 data_description.append(item['description'])

data_title
data_description

data_df = pd.DataFrame({'title':data_title, 
                        'description':data_description})
#%% 

data_df['title'] = data_df['title'].apply(
    lambda x : re.sub(r'[^ ㄱ-ㅣ가-힣]+', " ", x))

data_df['description'] = \
data_df['description'].apply(
    lambda x : re.sub(r'[^ ㄱ-ㅣ가-힣]+', " ", x))

data_df.head()

data_title_tfidf = tfidf.transform(data_df['title'])

data_title_predict = SA_lr_best.predict(data_title_tfidf)

data_df['title_label'] = data_title_predict

data_description_tfidf = tfidf.transform(data_df['description'])

data_description_predict = SA_lr_best.predict(data_description_tfidf)

data_df['description_label'] = data_description_predict

#%%

data_df.to_csv(file_name+'.csv', encoding = 'euc-kr')

data_df.head()
print(data_df['title_label'].value_counts())
print(data_df['description_label'].value_counts())

#%%

columns_name = ['title', 'title_label', 
                'description', 'description_label']
NEG_data_df = pd.DataFrame(columns = columns_name)
POS_data_df = pd.DataFrame(columns = columns_name)

for i, data in data_df.iterrows():
    title = data["title"]
    description = data["description"]
    t_label = data["title_label"]
    d_label = data["description_label"]

    if d_label == 0: 
        NEG_data_df = \
        NEG_data_df.append(pd.DataFrame([[title, t_label, 
                                          description, d_label]],
                         columns = columns_name), ignore_index = True)

    else : 
        POS_data_df = \
        POS_data_df.append(pd.DataFrame([[title, t_label, 
                                          description, d_label]],
                         columns = columns_name), ignore_index = True)


NEG_data_df.to_csv('data/'+file_name+'_NES.csv', encoding = 'euc-kr')
POS_data_df.to_csv('data/'+file_name+'_POS.csv', encoding = 'euc-kr')

len(NEG_data_df), len(POS_data_df)

#%% 결과 시각화

POS_description = POS_data_df['description']

POS_description_noun_tk = []
for d in POS_description:

    POS_description_noun_tk.append(okt.nouns(d))   

POS_description_noun_join = []
for d in POS_description_noun_tk:

   d2 = [w for w in d if len(w) > 1] 

   POS_description_noun_join.append(" ".join(d2))  

NEG_description = NEG_data_df['description']

NEG_description_noun_tk = []
NEG_description_noun_join = []

for d in NEG_description:

    NEG_description_noun_tk.append(okt.nouns(d)) 

for d in NEG_description_noun_tk:

    d2 = [w for w in d if len(w) > 1] 

    NEG_description_noun_join.append(" ".join(d2))  

POS_tfidf = TfidfVectorizer(tokenizer = okt_tokenizer, min_df = 2)
POS_dtm = POS_tfidf.fit_transform(POS_description_noun_join)

POS_vocab = dict()
for idx, word in enumerate(POS_tfidf.get_feature_names()):
    POS_vocab[word] = POS_dtm.getcol(idx).sum()

POS_words = \
sorted(POS_vocab.items(), key = lambda x: x[1], reverse = True)

NEG_tfidf = TfidfVectorizer(tokenizer = okt_tokenizer, min_df = 2)
NEG_dtm = NEG_tfidf.fit_transform(NEG_description_noun_join)

NEG_vocab = dict()
for idx, word in enumerate(NEG_tfidf.get_feature_names()):
    NEG_vocab[word] = NEG_dtm.getcol(idx).sum()

NEG_words = \
sorted(NEG_vocab.items(), key = lambda x: x[1], reverse = True)

#%%

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_location = 'C:/Windows/Fonts/malgun.ttf'
font_name = fm.FontProperties(fname = font_location).get_name()
matplotlib.rc('font', family = font_name)

max = 15  

plt.bar(range(max), [i[1] for i in POS_words[:max]], color = "red")
plt.title("긍정 뉴스의 단어 상위 %d개" %max, fontsize = 15)
plt.xlabel("단어", fontsize = 12)
plt.ylabel("TF-IDF의 합", fontsize = 12)
plt.xticks(range(max), [i[0] for i in POS_words[:max]], rotation = 70)
plt.show()

plt.bar(range(max), [i[1] for i in NEG_words[:max]], color = "blue")
plt.title("부정 뉴스의 단어 상위 %d개" %max, fontsize = 15)
plt.xlabel("단어", fontsize = 12)
plt.ylabel("TF-IDF의 합", fontsize = 12)
plt.xticks(range(max), [i[0] for i in NEG_words[:max]], rotation = 70)
plt.show()

