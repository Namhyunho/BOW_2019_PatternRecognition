# BOW_2019_PatternRecognition


---
### References

#### [이론적인 내용과 코드 참고](https://github.com/TrungTVo/spatial-pyramid-matching-scene-recognition)
---

### leve_0에 PCA와 LDA를 적용하여 성능 계선시도.

leve_0의 성능 43.8%

#### 1. 히스토그램 단에 LDA와 PCA를 적용시켜 보았다.

 - LDA의 경우 30% 의 성능이 나왔다.
   - code
   
   ```
   from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
   
   lda = LinearDiscriminantAnalysis(n_components=8)
   train_hist_Lda = lda.fit(train_hist, train_labels).transform(train_hist)
   test_hist_Lda = lda.transform(test_hist)
   
   clf = LinearSVC(random_state=0, C=0.1)

   clf.fit(train_hist_Lda, train_labels)

   predict = clf.predict(test_hist_Lda)

   ```

 - PCA의 경우 5% 의 성능이 나왔다.
    - code
   ```
   from sklearn.decomposition import PCA
   pca = PCA(n_components=8)
   train_hist_pca = pca.fit(train_hist).transform(train_hist)
   test_hist_pca = pca.fit(test_hist).transform(test_hist)
   
   ```
---
LDA 히스토그램의 특징들의 적절하게 사형 시켜 조금더 특징을 구분 짖는방법으로 적용해봄. 
PCA 히스토그렘의 노이즈를 줄이는 하나의 방법으로 적요해봄.
#### 2. kmeans이전에 PCA를 적용 시켜 보았다.

 - PCA의 경우 21% 의 성능이 나왔다.
    - code
   ```
   from sklearn.decomposition import PCA
   pca = PCA(n_components=20)
   all_train_append = pca.fit(all_train_append).transform(all_train_append)
   
   ```
---
이전의 시도에서 PCA의 성능이 너무 낮게 나와 sift로 뽑은것들의 노이즈를 제거하기 위한 방법으로 적용해봄.


### 결론

- 성능이 계선될것 같았던 방법들이 좋은 성능을 얻지못해 안타깝게 느껴짐.
   
 
