# record

### [car_number.ipynb](https://github.com/vacker92/record/blob/main/car_number.ipynb)
> 자동차 번호판 인식기 이미지 프로세싱
- [x] 작업 목록
- cv2 이미지 불러오기
- GaussianBlur 노이즈 제거
- findContours 이미지 윤곽선 찾기
- boundingRect 윤곽선을 감싸는 사각형 구하기
- rectangle 이미지에 사각형 그리기
- possible_contours 번호판 후보들 추려내기
- np.linalg.norm(a - b) 벡터 a와 벡터 b 사이의 거리 구하기 
- np.arctan() 아크탄젠트 값 구하기, np.degrees() 라디안을 도(º)로 변경
- copyMakeBorder() 이미지 패딩
- pytesseract.image_to_string 이미지에서 글자 읽기

### [yolov5_youtube.py](https://github.com/vacker92/record/blob/main/yolov5_youtube.py)  
[결과](https://www.youtube.com/watch?v=1Q0Q_CRRh2c)  
> yolov5를 사용해 동영상 객체인식
- [x] 작업 목록
- pafy 라이브러리 -> 유투브 동영상 크롤링
- get_video_from_url 함수 : url에서 새 비디오 스트리밍 객체를 생성
- load_model 함수 : PyTorch 허브에서 제공하는 모델 (YOLOv5 로드)

### [gas.ipynb](https://github.com/vacker92/record/blob/main/gas.ipynb)
[가스 공급량 데이터 출처](https://www.data.go.kr/data/15091497/fileData.do)  
[기온 데이터 출처](https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36) 
> 가스공급량 시계열 예측 
- [x] 작업 목록
- 온도추가 전과 추가 후 상관관계 분석
- tenserfolw LinearRegression 비교
- 다양한 모델들로 예측값 확인
- lstm을 사용해 시계열 예측 

### [imagecrawling.py](https://github.com/vacker92/record/blob/main/imagecrawling.py)
> selenium을 이용한 구글 이미지 크롤링 

### [naver_api_news.py](https://github.com/vacker92/record/blob/main/naver_api_news.py)
> naver api를 활용해 뉴스기사 크롤링
