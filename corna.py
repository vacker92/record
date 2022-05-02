from urllib.request import Request, urlopen
from urllib.parse import urlencode,unquote,quote_plus
import xmltodict
import pandas as pd
import matplotlib.pyplot as plt


# 호출 URL주소
url = 'http://openapi.data.go.kr/openapi/service/rest/Covid19/getCovid19InfStateJson'

# 서비스키 등록
ServiceKey = 'kL0tcoxy%2FuxaWh43HpkDWw6NO7utthuKg2zV4YGJ%2Blsz561Q0mvvj3GvxfmcVH0wrEVoneMZdXTiQxNkl2P9jQ%3D%3D'

# 전달할 매개 변수 설정
startCreateDt = '20200405'      # 데이터 생성일 시작범위
endCreateDt = '20211218'        # 데이터 생성일 종료범위
queryParams = '?' + urlencode({ 
    quote_plus('ServiceKey') : ServiceKey,
    quote_plus('pageNo') : 1, 
    quote_plus('numOfRows') : 10,
    quote_plus('startCreateDt') : startCreateDt, 
    quote_plus('endCreateDt') : endCreateDt 
    })

# 오픈API 호출
request = Request(url + unquote(queryParams))
request.get_method = lambda: 'GET'
response_body = urlopen(request).read()

# XML 결과를 딕셔너리로 변환
result_dict = xmltodict.parse(response_body)

# NORMAL_SERVICE(정상)인 경우만 처리
if (result_dict['response']['header']['resultCode'] == '00'):
    # 딕셔너리 결과 중 모든 item 항목을 데이터프레임으로 전환
    df = pd.DataFrame(result_dict['response']['body']['items']['item'])
else:
    quit()

df.info()

# 누적 확진자수 정수화
df['decideCnt'] = df['decideCnt'].astype(int)

# 기준시간 문자열을 시간 객체로 변환
df['stateDt'] = pd.to_datetime(df['stateDt'])

# 기준시간 기준으로 재 정렬
# (요청된 최초 자료에 일부 날짜가 뒤 섞인 경우가 있음)
df = df.sort_values(by = 'stateDt', ascending = False)

# 일별 확진자수 계산
df['decideCntByDay'] = df['decideCnt'].diff(periods = -1)

# 마지막 행 삭제, 일별 확진자수가 NaN인 행
df.drop(df.index[-1], inplace = True)

# 자료 순서를 기준시간 날짜순(오름차순)으로 재 정렬
df = df[::-1]

# 주간(7일)/월간(30일) 이동평균값 계산
df['ma7'] = df['decideCntByDay'].rolling(window = 7).mean()
df['ma30'] = df['decideCntByDay'].rolling(window = 30).mean()

# 그래프 초기화
plt.rc('font', family = 'Malgun Gothic')
fig, ax = plt.subplots(figsize = (10, 4))

ax.scatter(df['stateDt'], df['decideCntByDay'], c = 'red', s = 3, alpha = 0.5, label = '일일확진자수')
ax.plot(df['stateDt'], df['ma7'], label = '주간(7일) 이동평균')
ax.plot(df['stateDt'], df['ma30'], label = '월간(30일) 이동평균')

ax.legend()
ax.set_title('코로나19 일별확진자수 이동평균 추세')
ax.set_xlabel('기간')
ax.set_ylabel('일별 확진자수')
ax.grid(axis = 'y')
ax.tick_params(right = True, labelright = True) # 우측 틱과 레이블 표시

plt.show()


df['createDt'] = pd.to_datetime(df['createDt'], format='%Y-%m-%d')
df = df.set_index('createDt')

kia_tmp = df[:'2021-12-18']
kia_tmp
kia_tmp.tail()

kia_trunc = pd.DataFrame({
    'ds': kia_tmp.index,
    'y': kia_tmp['decideCntByDay'] })


from fbprophet import Prophet

m = Prophet()
m.fit(kia_trunc)
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)

forecast.info()
m.plot(forecast[['ds','trend','weekly','yhat']]);
m.plot_components(forecast);

df.info()
plt.figure(figsize=(13,5))
plt.plot(df.index, df['DECIDE_CNT'], label='real')
plt.plot(forecast['ds'],forecast['yhat'], label='forecast')
# plt.savefig('corna.png')