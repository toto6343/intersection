## 우리인재개발원(우리컴퓨터아카데미)
```
교차로 교통 장애물 및 이벤트 감지 시스템 개발
```
## 우리인재 팀
깃허브 링크 : https://github.com/joyoungkyu
```
팀장 : 조영규 
팀원 : 김우혁, 유혜정, 정날빛
```
## 프로젝트 주제
```
교차로 돌발진입 포착
```
## 프로젝트 진행
![과정](https://github.com/user-attachments/assets/52282c9f-debb-4efe-a20b-0716b9a2539a)
### 데이터수집
![이미지](https://github.com/user-attachments/assets/4258d17c-e76c-4dfa-8044-170fd6362205)

야간 :
![야간 이미지](https://github.com/user-attachments/assets/8d9de2d7-6b4b-400b-b1e2-a745b9241d55)
```
참고자료 : https://docs.ultralytics.com/
https://www.its.go.kr/opendata/
https://www.roboflow.com/
```
### 데이터검증
자동라벨링 버튼사진 :
![4](https://github.com/user-attachments/assets/0c3a0d0a-7643-4566-ad50-7c0d7afb04a6)

자동라벨링 과정사진 :
![2](https://github.com/user-attachments/assets/d5c0e756-5b7b-4f9f-a70a-0fc25fb6ad71)

결과사진 :
![결과이미지](https://github.com/user-attachments/assets/81a2b7d4-60f2-4842-ad8c-a3c1fd8d2de8)

### 데이터 전처리
데이터셋 사진 :
![전처리](https://github.com/user-attachments/assets/b47fca84-3a7a-415a-afea-bceb33ee01bd)

비전 제작 :
![모델재작](https://github.com/user-attachments/assets/1f635a4f-5704-4bed-a875-f382f5aaa514)

```
Train : Vaild : Test =  8 : 1 : 1
```
### 모델 학습 및 튜닝
모델 코드 사진 :
![코드](https://github.com/user-attachments/assets/3f495da8-2da4-4a2d-b114-0d2b4d42c5ab)

훈련 결과 그래프 :

![그래프](https://github.com/user-attachments/assets/7e8cdeff-bbd8-47e9-90dc-6d29592f7c65)![그래프(상세)](https://github.com/user-attachments/assets/deb2a836-683c-4be6-8c74-af84123567cb)![클래스](https://github.com/user-attachments/assets/36b09dd6-dbda-42c4-925c-30ecbbcad6b1)![클래스1](https://github.com/user-attachments/assets/204e14a5-6b34-4062-bf91-21789c943a63)

컨피던스 메트릭스 결과(제작한 모델) :

![매트릭스](https://github.com/user-attachments/assets/d4edddb9-9286-405b-83d1-0f5c3f275b4d)![매트릭스 그래프](https://github.com/user-attachments/assets/c5c20532-bcb6-4dff-8e96-be0416c51f08)

### 모델 분석 및 검증
YOLO11모델 :
![캡처](https://github.com/user-attachments/assets/8609560a-b575-440c-a570-bb6bfdc36c3f)

실제 재작 모듈 :
![image](https://github.com/user-attachments/assets/561e2eea-0ecb-4c46-9024-fb199d85adcc)
```
둘다 실행 결과를 캡처한 것으로 둘의 차이점을 비교본으로 개시
```
### 모델 배포
실시간 영상 : PPT링크 참고

데시보드 연동 이미지 (직접 만든 모델) :

![데시보드(직접)](https://github.com/user-attachments/assets/d73a228f-5f1b-4e25-8f95-26225a98ee14) ![데시보드(직접)1](https://github.com/user-attachments/assets/7f9abf32-095b-4ab0-bed9-9ddeea53eb9c)
![데시보드(직접)2](https://github.com/user-attachments/assets/6911bd92-3579-4ca8-90e7-784ceeae8012) ![데시보드(직접)3](https://github.com/user-attachments/assets/0d9a7d00-6437-4919-8731-bf8c662830f3) ![데시보드(직접)4](https://github.com/user-attachments/assets/471f51b3-4b31-4b3e-9b9f-c3404b174ee4) ![데시보드(직접)5](https://github.com/user-attachments/assets/9faed452-e7e1-4960-b3de-40c591ee4a0b) ![데시보드(직접)6](https://github.com/user-attachments/assets/ede6eb8e-f848-47ed-bc59-9099ac542b57)

코드 (직접 만든 모델) :

https://github.com/JoYoungKyu/team_project/blob/main/dashboard/violation_dashboard2.py

코드 (YOLO11n.pt모델) :

https://github.com/JoYoungKyu/team_project/blob/main/dashboard/app.py

데시보드 연동 이미지 (YOLO11n.pt모델) :

![대시보드](https://github.com/user-attachments/assets/0992c478-e752-47b0-9a0b-c97fc2441b88)

텐서보드 (직접 만든 모델) :

![2025-04-23 14 03 09](https://github.com/user-attachments/assets/22aff234-3694-44ec-bf76-9141b3318260) ![텐서보드](https://github.com/user-attachments/assets/c4da8d93-465d-44da-afca-3db88e5bb1ae)

### 피드백
```
컴퓨전매트릭스 시도(목요일에 예정),초기모델보다 더 정확하게 훈련시킨 모델사용, 영역지정은 제외, 벡터값을 기준으로 불법유턴같은 돌발상황 포착 사람과 차량의 거리측정해서 가까우면 위험 알리기 (경고음) 이용
```
알림 화면 :

![2025-04-23 13 59 55](https://github.com/user-attachments/assets/d13d90aa-52dc-4374-a794-946c061a9724)

### 향후계획
1. 인도 및 차도 구분
2. 차량과 차량 사이의 거리 계산 및 알림 서비스
3. 여러 종류의 차량 데이터 수집 및 가공
4. 여러 종류의 동물 데이터 수집 및 가공
5. 라벨링 신뢰도 조정

## PPT 자료
JoyK_교차로 돌발 진입 감지.pdf
