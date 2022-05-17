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
