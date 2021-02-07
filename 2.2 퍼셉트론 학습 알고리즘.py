import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt             ....?

class Perceptron(object):
    '''
        eta : 학습률
        n_iter : 반복 수
    '''
    def __init__(self, eta=0.01, n_iter=50, random_state=1):    # Perceptron 객체 초기화
        self.eta            = eta
        self.n_iter         = n_iter
        self.random_state   = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)         # 난수 생성, seed = 1
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size= 1 + X.shape[1])     # 가중치 (X.shape[1]:데이터셋 특성 수), +1은 절편 (표준 편차가 0.01인 정규 분포에서 뽑은 랜덤한 작은 수)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))     # 학습률 * (real_y.value - predict_y.value)
                self.w_[1:] += update * xi                          # 가중치 변화량 = 학습률*(real_y.value - pred_y.value)*xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):                                     # 벡터 점곱 계산
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):                                       # 예측
        return np.where(self.net_input(X) >= 0.0, 1, -1 )       # 0.0 보다 크면 1 작으면 -1


df = pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', header=None)
# setosa와 versicolor를 선택합니다
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# 꽃받침 길이와 꽃잎 길이를 추출합니다
X = df.iloc[0:100, [0, 2]].values

# 산점도를 그립니다
# plt.scatter(X[:50, 0], X[:50, 1],
#             color='red', marker='o', label='setosa')
# plt.scatter(X[50:100, 0], X[50:100, 1],
#             color='blue', marker='x', label='versicolor')
#
# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')
# plt.legend(loc='upper left')
#
# plt.show()