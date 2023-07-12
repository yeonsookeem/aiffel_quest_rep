# AIFFEL Campus Online 5th Code Peer Review Templete
- 코더 : 조대호
- 리뷰어 : 김연수


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [O] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  
- [O] 주석을 보고 작성자의 코드가 이해되었나요?
  > 주석이 상세하게 적혀있어, 코드 이해에 도움이 되었습니다.
- [O] 코드가 에러를 유발할 가능성이 없나요?
  >에러 없이 모두 작동하는 것으로 보입니다.
- [O] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 주석의 내용으로 보아, 전체 코드에 대한 충분한 이해를 바탕으로 작성된 것 같습니다.
- [O] 코드가 간결한가요?
  > 전반적으로 코드가 간결하여 이해하는 데에 어려움이 없었습니다.

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.
```python

# LGBM
from sklearn.model_selection import GridSearchCV

#param_grid에 탐색할 xgboost 관련 하이퍼 파라미터를 넣어서 준비
param_grid = {
    'n_estimators': [80,90, 100,110],
    'max_depth': [8,9,10],
    'learning_rate':[0.01,0.1,0.05],
}
     
#모델 준비(예시에서는 lightgbm 사용)
from lightgbm import LGBMRegressor
random_state = 2000
model = LGBMRegressor(random_state=random_state)

#다양한 하이퍼 파라미터를 넣어서 학습을 시킨다.
grid_model = GridSearchCV(model, param_grid=param_grid, \
                        scoring='neg_mean_squared_error', \
                        cv=5, verbose=1, n_jobs=5)

grid_model.fit(train_1, y)

#필요한 정보만 추출
params = grid_model.cv_results_['params']
score = grid_model.cv_results_['mean_test_score']

#위에서 뽑은 max_depth,n_estimators,score 데이터프레임으로 묶기
scores = list(score)
for i in range(len(params)):
    params[i]['score'] = scores[i]

results = pd.DataFrame(params)

#rmse 추가
results['RMSLE'] = np.sqrt(-1 * results['score'])
results

#위의 결과를 바탕으로 파라미터를 재조정하여 훈련
model = LGBMRegressor(learning_rate=0.1, max_depth=9, n_estimators=110, random_state=random_state)
model.fit(train_1, y)
prediction = model.predict(test_1)
prediction

#예측 결과에 np.expm1()을 씌워서 다시 원래 스케일로 되돌리자
prediction = np.expm1(prediction)

#submission 파일에 price 값 저장
```

# 참고 링크 및 코드 개선
```python
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
LGBM  https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html  
참고자료 https://yssa.tistory.com/entry/Big-Data-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A0%84%EC%B2%98%EB%A6%AC
