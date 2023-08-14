# AI

## 진단모델 - KoBERT

사용자로부터 입력받은 증상으로부터 질병명을 예측합니다.

### 1. 학습 데이터셋 (*출처: 하이닥)

<img src="/img/csv.png" alt="csv" style="width: 50%;">

약 4800개의 데이터를 학습시켜 다음과 같은 질병을 예측할 수 있습니다.


'ADHD', 'A형 간염', '각막염', '감기', '건선', '결핵', '고혈압', '골다공증', '골절', '공황장애', '기흉',
 '당뇨병', '류마티스 관절염', '목 디스크', '방광염', '변비', '불면증', '비만', '비염', '빈혈', '성조숙증', '소화불량', '수족냉증', '식중독',
 '아토피 피부염', '안구건조증', '알코올중독증', '요로결석', '요실금', '우울증', '인플루엔자', '자궁근종', '장염','접촉성 피부염', 
 '조울증', '중이염', '질염', '충치', '치매', '치은염', '치질','통풍', '패혈증', '패렴', '협심증', '화상'
 
### 2. 정확도

### 3. Filter

* 한쪽 성별에서만 나타나는 질병이 존재, 필터링하는 과정을 거침
