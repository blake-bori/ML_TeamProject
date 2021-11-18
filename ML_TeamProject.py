import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, precision_score, recall_score, f1_score, \
    roc_auc_score, roc_curve, RocCurveDisplay, silhouette_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier
import plotly.express as px
from imblearn.over_sampling import SMOTE


def dataVisualize(data):
    # data : 시각화할 데이터

    # 데이터셋의 attribute별 분포도 확인(그래프)
    # attribute에 존재하는 값의 수가 적으면 원그래프로 표현하는 것도 괜찮을듯 / 많으면 histogram
    # 어느 한 attribute의 특정 target값을 가진 데이터의 수를 histogram으로 표현
    # 등등 데이터를 잘 표현

    # 텍스트로 데이터 표현 (전체 데이터, 데이터 feature별로의 데이터 수, nan값 수)
    print(df.head(10))
    print(df.info())
    print(df.isna().sum())

    # 탑승자에 대한 정보
    dataColumns1 = ['destination', 'passanger', 'weather', 'temperature', 'time', 'gender', 'age', 'maritalStatus',
                    'has_children', 'income']
    # 쿠폰 정보
    dataColumns2 = ['coupon', 'expiration']
    # 탑승자 기준 쿠폰 가게 위치에 대한 정보
    dataColumns3 = ['Bar', 'CoffeeHouse', 'CarryAway', 'RestaurantLessThan20', 'Restaurant20To50']
    # 탑승자 기준 방향 정보
    dataColumns4 = ['direction_same', 'direction_opp']

    # 탑승자에 대한 정보
    for col in dataColumns1:
        sns.countplot(y=data[col], hue=data["Y"])
        plt.subplots_adjust(left=0.225 if col == "maritalStatus" or col == "income" else 0.2)
        plt.xlabel("count")
        plt.ylabel("")
        plt.title(col)
        plt.show()

    # 쿠폰 종류
    couponCnt = data["coupon"].value_counts()
    couponList = list(couponCnt.index)
    plt.pie(couponCnt, labels=couponList, autopct='%.1f%%')
    plt.title("coupon type")
    plt.show()

    # 쿠폰에 대한 정보
    for col in dataColumns2:
        sns.countplot(y=data[col], hue=data["Y"])
        plt.subplots_adjust(left=0.27 if col == "coupon" else 0.1)
        plt.xlabel("count")
        plt.ylabel("")
        plt.title(col)
        plt.show()

    # 쿠폰 카게에 대한 정보
    for col in dataColumns3:
        sns.countplot(y=data[col], hue=data["Y"])
        plt.xlabel("count")
        plt.ylabel("")
        plt.title(col)
        plt.show()

    # 탑승자의 방향에 대한 정보
    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    axes = axes.flatten()

    for ax, col in zip(axes, dataColumns4):
        sns.countplot(y=col, data=data, ax=ax, hue=data["Y"])
        plt.subplots_adjust(left=0.05, right=0.97)
        ax.set_xlabel("count")
        ax.set_ylabel("")
        ax.set_title(col)
    plt.show()

    # return 값 : 없음


def preprocessing(df, isCluster):
    # drop the 'car' column
    df = df.drop(['car'], axis=1)

    # Drop the row with the missing value.
    df = df.dropna(axis=0, how='any')

    # 인코딩 하는 feature
    label_feature = ['temperature', 'time', 'expiration', 'age', 'education', 'income', 'Bar', 'CoffeeHouse',
                     'CarryAway', 'RestaurantLessThan20', 'Restaurant20To50']
    onehot_feature = ['destination', 'passanger', 'weather', 'gender', 'maritalStatus', 'education',
                      'coupon', 'occupation']
    integration_feature = ['toCoupon_GEQ5min', 'toCoupon_GEQ15min', 'toCoupon_GEQ25min']

    # 인코딩 안하는 feature
    origin_feature = ["has_children", "direction_same"] if isCluster == True else ["has_children", "direction_same",
                                                                                   "Y"]
    # label encoding
    label = LabelEncoder()
    df_1 = df[label_feature]

    def label_encoder(data):
        for i in label_feature:
            data.loc[:, i] = label.fit_transform(data.loc[:, i])

            # 인코딩 되 값의 원래 값 보는 것
            # for k, l in enumerate(label.classes_): print(k, '->', l)
        return data

    X_label = label_encoder(df_1)
    X_label[origin_feature] = df[origin_feature]

    # onehot encoding
    X_onehot = pd.get_dummies(df[onehot_feature])

    # toCoupon feature
    x = df[integration_feature]
    toCoupon = x.sum(axis=1)

    # concat X_label, X_onehot
    cleaned_df = pd.concat([X_label, X_onehot, toCoupon], axis=1)
    cleaned_df.rename(columns={0: "toCoupon"}, inplace=True)

    if isCluster == True:
        # 클러스터링이라면 PCA로 feature수를 줄임
        pca = PCA(n_components=2)
        pcaData = pca.fit_transform(cleaned_df)
        scatterData = pd.DataFrame(pcaData, columns=["Component1", "Component2"])
        label = pd.DataFrame(df["Y"].reset_index(), columns=["Y"])
        scatterData = pd.concat([scatterData, label], axis=1)

        return scatterData

    else:
        return cleaned_df


# 스케일링 된 X_train과 scaler오브젝트를 반환합니다
# scaler는 string 입니다.
# 'StandardScaler', 'MinMaxScaler', 'RobustScaler' 사용 가능
def getScalered(X_train, scaler_name: str, scaled_cols: list):
    if scaler_name.lower() == 'standardscaler':
        scaler = StandardScaler()
    elif scaler_name.lower() == 'minmaxscaler':
        scaler = MinMaxScaler()
    elif scaler_name.lower() == 'robustscaler':
        scaler = RobustScaler()

    X_train_scaled = X_train.copy()

    scaler.fit(X_train_scaled[scaled_cols])
    X_train_scaled[scaled_cols] = scaler.transform(X_train_scaled[scaled_cols])

    return X_train_scaled, scaler


def classificationFunc(data, scalers, scaled_cols, models_params, cv):
    # data : 데이터셋(전처리 완료한 상태여야함)
    # scalers : 스케일러 '이름' 리스트
    # scaled_cols : 스케일링할 컬럼
    # models_params : { '모델' : { '파라미터' : 파라미터값 리스트 } } 형태의 dictionary
    # score : 최고의 파라미터를 찾기 위한 기준(classification은 accuracy)
    # cv : K-Fold의 K값

    # 데이터셋을 feature, target으로 나눔
    # 훈련 set과 테스트 set으로 나눈다. 8:2
    y = data['Y']
    X = data.drop(['Y'], axis=1)

    # y값이 1에 비해 0이 너무 적어 학습 결과 1로 치우지는 경향이 생겨 오버샘플링함
    smote = SMOTE(random_state=5)
    X, y = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=5, stratify=y)

    # 테스트용 (보여주기 용 데이터) : 한 사람의 정보를 쿠폰별로 복사 -> 학습된 모델에 적용시켰을 때 Y가 1인 쿠폰 종류를 반환
    recommend_test = X_test.iloc[0, :]
    recommend_test_df = [recommend_test.copy(), recommend_test.copy(), recommend_test.copy(), recommend_test.copy(),
                         recommend_test.copy()]
    recommend_test_df = pd.DataFrame(recommend_test_df)
    recommend_test_df = recommend_test_df.reset_index(drop="index")

    # column 36 ~ 40이 쿠폰
    for index, row in recommend_test_df.iterrows():
        for i in range(36, 41):
            if i == 36 + index:
                row[i] = 1
            else:
                row[i] = 0

    # 각 모델이 scaler별로 훈련된 결과를 저장(scaler별 최고의 파라미터임.)
    # { '모델' : {'스케일러':[모델_인스턴스, 파라미터_dict, 스코어, 스케일러_인스턴스]} }
    best_model_in_scaler: dict = {}
    # 각 모델이 가장 높은 스코어를 가질 때의 결과 저장
    # { '모델' : [모델_인스턴스, 파라미터_dict, 스코어, '스케일러', 스케일러_인스턴스] }
    best_models: dict = {}

    # 각 스케일러별 각 모델들의 최고의 파라미터를 찾아 저장한다.
    for scaler in scalers:
        # 학습 데이터를 스케일링 한다
        X_train_scaled = X_train
        X_train_scaled[scaled_cols], scaler_obj = getScalered(X_train, scaler, scaled_cols)
        X_test_scaled = X_test
        X_test_scaled[scaled_cols] = scaler_obj.transform(X_test_scaled[scaled_cols])

        # 모델 리스트에 있는 모델별로 최적의 파라미터를 찾아낸다
        for model in models_params:
            if model not in best_model_in_scaler: best_model_in_scaler[model] = {}

            print(scaler, "+", model, " , ", models_params[model], " , ", cv)
            best_model_in_scaler[model][scaler] = findBestParam(X_train_scaled, y_train, model, models_params[model],
                                                                cv)
            best_model_in_scaler[model][scaler].append(scaler_obj)

    # 각 모델의 가장 높은 스코어를 출력한 스케일러만을 저장한다
    for model in best_model_in_scaler:
        best_score: int = 0
        best_scaler = None  # 스케일러 오브젝트
        for scaler in best_model_in_scaler[model]:
            # best_model_in_scaler[model][scaler][2]는 스코어에 해당한다
            if best_model_in_scaler[model][scaler][2] > best_score:
                best_score = best_model_in_scaler[model][scaler][2]
                best_scaler = scaler

        best_model_best_scaler: list = best_model_in_scaler[model][best_scaler]
        # { '모델' : [모델_인스턴스, 파라미터_dict, 스코어, '스케일러', 스케일러_인스턴스] }
        best_models[model] = [best_model_best_scaler[0], best_model_best_scaler[1], best_model_best_scaler[2],
                              best_scaler,
                              best_model_best_scaler[3]]

    print("================================================================================================")
    print("================================================================================================\n")
    print("최종 결과")

    # 각 모델의 예측 결과 출력
    for model in best_models:
        print("모델 정보 : ", best_models[model][0])
        best_scaler_obj = best_models[model][4]

        X_train_scaled = X_train

        X_test_scaled = X_test
        recommend_test_df[scaled_cols] = best_scaler_obj.transform(recommend_test_df[scaled_cols])

        model_obj = best_models[model][0]
        model_obj.fit(X_train_scaled, y_train)

        couponList = ['Restaurant(<20)', 'Coffee House', 'Carry out & Take away', 'Bar', 'Restaurant(20-50)']
        test_pred=model_obj.predict(recommend_test_df)

        recommend_coupon=[]
        for i in range(0,5):
            if test_pred[i]==1:
                recommend_coupon.append(couponList[i])

        print(test_pred)
        print(recommend_coupon)

        print("Accuracy : ", model_obj.score(X_test_scaled, y_test))
        printConfusionMatrix(X_test_scaled, y_test, model_obj)
        printROC(X_test_scaled, y_test, model, model_obj)


def printConfusionMatrix(X_test_scaled, y_test, model_obj):
    print('=' * 50)
    # visualize the confusion matrix
    label = ['non-accept', 'accept']
    y_pred = model_obj.predict(X_test_scaled)
    print(X_test_scaled)
    cf = confusion_matrix(y_test, y_pred)
    print(cf)

    plot = plot_confusion_matrix(model_obj,
                                 X_test_scaled, y_test,
                                 display_labels=label,
                                 normalize='true')
    plot.ax_.set_title('Confusion Matrix')
    plt.show()

    pre = precision_score(y_test, y_pred)
    re = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print('+' * 30)
    print(f'precision : {pre}')
    print(f'recall : {re}')
    print(f'f1 score : {f1}')
    print('=' * 50)


def printROC(X_test_scaled, y_test, model_name: str, model_obj):
    if model_name.lower() == 'logisticregression':
        lr_probs = model_obj.predict_proba(X_test_scaled)
        # 1이 될 확률
        lr_probs = lr_probs[:, 1]
        # calculate scores
        lr_auc = roc_auc_score(y_test, lr_probs)
        # summarize scores
        print('Logistic: ROC AUC=%.3f' % (lr_auc))
        # calculate roc curves
        lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
        # plot the roc curve for the model
        plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
    elif model_name.lower() == 'randomforestclassifier':
        ax = plt.gca()
        rfc_disp = RocCurveDisplay.from_estimator(model_obj, X_test_scaled, y_test, ax=ax, alpha=0.8)
        rfc_disp.plot(ax=ax, alpha=0.8)
    elif model_name.lower() == 'decisiontreeclassifier':
        ax = plt.gca()
        rfc_disp = RocCurveDisplay.from_estimator(model_obj, X_test_scaled, y_test, ax=ax, alpha=0.8)
        rfc_disp.plot(ax=ax, alpha=0.8)

    plt.show()


def clusteringFunc(data, scalers, scaled_cols, models_params, cv):
    # data : 데이터셋(전처리 완료한 상태여야함)
    # scaler : 스케일러 '이름' 리스트
    # scaled_cols : 스케일링할 컬럼
    # model_params : { '모델' : { '파라미터' : 파라미터값 리스트 } } 형태의 dictionary
    # score : 최고의 파라미터를 찾기 위한 기준(clustering은 silouette)
    # cv : K-Fold의 K값 (클러스터링은 필요할까?)

    # scaler와 model은 string 값

    # 데이터셋을 feature, target으로 나눔
    # 훈련 set과 테스트 set으로 나눈다. 8:2
    y = data['Y']
    X = data.drop(['Y'], axis=1)

    smote = SMOTE(random_state=5)
    X, y = smote.fit_resample(X, y)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000, stratify=y)

    # 각 모델이 scaler별로 훈련된 결과를 저장(scaler별 최고의 파라미터임.)
    # { '모델' : {'스케일러':[모델_인스턴스, 파라미터_dict, 스코어, 스케일러_인스턴스]} }
    best_model_in_scaler: dict = {}
    # 각 모델이 가장 높은 스코어를 가질 때의 결과 저장
    # { '모델' : [모델_인스턴스, 파라미터_dict, 스코어, '스케일러', 스케일러_인스턴스] }
    best_models: dict = {}

    # 각 스케일러별 각 모델들의 최고의 파라미터를 찾아 저장한다.
    for scaler in scalers:
        # 학습 데이터를 스케일링 한다
        X_train_scaled = X
        X_train_scaled[scaled_cols], scaler_obj = getScalered(X, scaler, scaled_cols)
        # 모델 리스트에 있는 모델별로 최적의 파라미터를 찾아낸다
        for model in models_params:
            if model not in best_model_in_scaler: best_model_in_scaler[model] = {}

            best_model_in_scaler[model][scaler] = findBestParam(X_train_scaled, None, model, models_params[model], cv)
            best_model_in_scaler[model][scaler].append(scaler_obj)

    # 각 모델의 가장 높은 스코어를 출력한 스케일러만을 저장한다
    for model in best_model_in_scaler:
        best_score: int = 0
        best_scaler = None  # 스케일러 오브젝트
        for scaler in best_model_in_scaler[model]:
            # best_model_in_scaler[model][scaler][2]는 스코어에 해당한다
            if best_model_in_scaler[model][scaler][2] > best_score or best_score == 0:
                best_score = best_model_in_scaler[model][scaler][2]
                best_scaler = scaler

        best_model_best_scaler: list = best_model_in_scaler[model][best_scaler]
        # { '모델' : [모델_인스턴스, 파라미터_dict, 스코어, '스케일러', 스케일러_인스턴스] }
        best_models[model] = [best_model_best_scaler[0], best_model_best_scaler[1], best_model_best_scaler[2],
                              best_scaler, best_model_best_scaler[3]]

    print("================================================================================================")
    print("최종 결과")

    # 각 모델의 예측 결과 출력
    for model in best_models:
        print("모델 정보 : ", best_models[model][0])
        print("Silhouette Score : ", best_models[model][2])

        model_obj = best_models[model][0]
        model_obj.fit(X)
        y_pred = model_obj.labels_

        printPurity(y, y_pred)
        printScatter(X, y_pred)


def cv_silhouette_scorer(estimator, X):
    estimator.fit(X)
    cluster_labels = estimator.labels_
    num_labels = len(set(cluster_labels))
    num_samples = len(X.index)
    if num_labels == 1 or num_labels == num_samples:
        return -1
    else:
        return silhouette_score(X, cluster_labels)


def printPurity(y_test, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_test, y_pred)

    # print purity score
    purity_sco = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
    print("Purity score: %.3f" % purity_sco, "\n")


def printScatter(X_test_scaled, y_pred):
    fig = px.scatter(X_test_scaled,x="Component1",y='Component2',color=y_pred)
    fig.show()


def findBestParam(x_train, y_train, model, model_params, cv):
    # x_train : train 데이터의 feature
    # y_train : train 데이터의 target (clustering은 None을 받음)
    # model : 모델
    # model_params : { 파라미터 : 파라미터값 리스트 } 형태의 dictionary
    # score : 최고의 파라미터를 찾기 위한 기준(classification은 accuracy, clustering은 silouette)
    # cv : K-Fold의 K값 (클러스터링은 필요할까?)

    # 정확도가 크게 안오를때까지 반복(while)
    #   파라미터별로 반복 (for param in model_params)
    #       파라미터의 값을 조정하면서 정확도(accuracy, silhouette)를 계산
    #   파라미터별로 한바퀴 돌았을 때 정확도 변화 없으면 while반복 종료

    # 파라미터별로 변화값
    # 카테고리컬 값은 변화x

    parameterInterval = {"max_depth": 1, "max_leaf_nodes": 1, "min_samples_leaf": 1, "min_samples_split": 1,
                         "C": 0.01, "max_iter": 100,
                         "max_samples": 1, "n_estimators": 5,
                         "n_clusters": 1, "n_init": 1,
                         "eps": 0.01, "min_samples": 1,
                         "degree": 1, "n_components": 1, "n_neighbors": 2, "gamma": 0.1}

    classificationModelList = ["logisticregression", "decisiontreeclassifier", "randomforestclassifier"]
    clusteringModelList = ["kmeans", "dbscan", "spectralclustering"]

    if model.lower() in classificationModelList:
        index = classificationModelList.index(model.lower())
        initModel = LogisticRegression() if index == 0 else DecisionTreeClassifier() if index == 1 else RandomForestClassifier()
    elif model.lower() in clusteringModelList:
        index = clusteringModelList.index(model.lower())
        initModel = KMeans() if index == 0 else DBSCAN() if index == 1 else SpectralClustering()
    else:
        initModel = None
        print("Error : Model Name Invalid")

    kFold = KFold(n_splits=cv, shuffle=True, random_state=10)

    # y_train이 None이 아니다 == classification이다
    if y_train is not None:
        # 그리드 서치로 파라미터 리스트 중에서 가장 괜찮은 파라미터를 찾고 그 파라미터의 값을 조금씩 조정하면서 최종 파라미터를 구함
        # 그리드 서치로 파라미터 리스트 중 어떤 값들을 선택할지 고름
        print("그리드 서치로 최대한 global maximum에 가깝운 파라미터 찾는 중...")
        grid_search = GridSearchCV(initModel, param_grid=model_params, cv=kFold, n_jobs=4)
        grid_search.fit(x_train, y_train)
        print("그리드 서치 결과")
        print("model : ", initModel)
        print("score : ", grid_search.best_score_)
        print("parameter : ", grid_search.best_params_)
        print()

        bestResult = [initModel, grid_search.best_params_, grid_search.best_score_]
        initModel.set_params(**grid_search.best_params_)

        changeParameters = bestResult[1]
        while True:
            curResult = bestResult.copy()
            curModel = curResult[0]
            print("----------------------------------------------------------------------------------------")
            print("반복 시작")
            print("정확도 : ", curResult[2])
            print()

            # 파라미터별로 값을 조정하면서 정확도 끌어올림
            for param in changeParameters:
                # 파라미터 값을 바꿔도 정확도가 안오를때까지 반복
                parameterMove = True  # 파라미터 값을 올릴지 말지 (True면 올린다, False면 내린다)
                parameterMoveStk = 2
                print(param, "의 값 : ", changeParameters[param])
                print("변화 폭 : ", parameterInterval[param] if param in parameterInterval else 0)
                print(param, "의 값 조정 시작")
                while True:
                    # 값을 올리거나 내리고 파라미터 재설정
                    # 값을 올리면서 정확도 올림 올려도 정확도가 안오르면 파라미터를 내려보고 내려도 안오르면 반복 종료
                    if parameterMove == True:
                        if param in parameterInterval:
                            print(param, "의 파라미터 값 증가시킴(", changeParameters[param], "->",
                                  changeParameters[param] + parameterInterval[param], ")")
                            changeParameters[param] += parameterInterval[param]
                        else:
                            print(param, "값 조정 종료")
                            print()
                            break
                    else:
                        if param in parameterInterval:
                            print(param, "의 파라미터 값 감소시킴(", changeParameters[param], "->",
                                  changeParameters[param] - parameterInterval[param], ")")
                            changeParameters[param] -= parameterInterval[param]

                    curModel.set_params(**changeParameters)
                    # 파라미터가 재설정된 모델을 다시 학습시킴
                    tempScore = cross_val_score(curModel, x_train, y_train, cv=kFold).mean()  # 실루엣 스코어로 결정
                    if curResult[2] < tempScore:
                        print("정확도 오름 : ", curResult[2], "->", tempScore, "\n")
                        curResult[1] = changeParameters
                        curResult[2] = tempScore
                        parameterMoveStk = 2
                    elif curResult[2] == tempScore and parameterMoveStk > 0:
                        print("정확도 변화 없음 : ", parameterMoveStk, "번 더 시도")
                        parameterMoveStk -= 1
                        continue
                    else:
                        if parameterMove == True:
                            print("정확도 안오름 반대방향으로 진행", "\n")
                            changeParameters[param] -= (3 - parameterMoveStk) * parameterInterval[param]
                            parameterMove = False
                            parameterMoveStk = 2

                        else:
                            print("정확도 안오름", param, "값 조정 종료")
                            print()
                            changeParameters[param] += (3 - parameterMoveStk) * parameterInterval[param]
                            break

            if bestResult[2] < curResult[2]:
                print("전체 파라미터 조정 후 정확도 상승 - 재반복")
                print()
                bestResult = curResult
            else:
                print("전체 파라미터 조정 후 정확도 상승 없음 - 반복 종료 값 반환")
                print()
                break

    ##############################################################################################################

    # y_train이 None이다 == clustering이다
    else:
        # 그리드 서치로 파라미터 리스트 중에서 가장 괜찮은 파라미터를 찾고 그 파라미터의 값을 조금씩 조정하면서 최종 파라미터를 구함
        # 그리드 서치로 파라미터 리스트 중 어떤 값들을 선택할지 고름
        # initModel=KMeans(init='random', max_iter= 500, n_clusters= 5).fit(x_train)
        # print("xxxx")
        # print(silhouette_score(x_train,initModel.labels_))
        #
        # bestResult = [initModel, initModel.get_params(), cv_silhouette_scorer(initModel,x_train)]
        # print("반복 전 초기 모델 결과 : ", bestResult)
        # print()
        print("그리드 서치로 최대한 global maximum에 가깝운 파라미터 찾는 중...")
        grid_search = GridSearchCV(initModel, param_grid=model_params, scoring=cv_silhouette_scorer, n_jobs=4)
        grid_search.fit(x_train)
        print("그리드 서치 결과")
        print("model : ", initModel)
        print("score : ", grid_search.best_score_)
        print("parameter : ", grid_search.best_params_)
        print()

        bestResult = [initModel, grid_search.best_params_, grid_search.best_score_]
        initModel.set_params(**grid_search.best_params_)

        changeParameters = bestResult[1]
        while True:
            curResult = bestResult.copy()
            curModel = curResult[0]
            print("----------------------------------------------------------------------------------------")
            print("반복 시작")
            print("초기 정확도 : ", curResult[2])
            print()

            # 파라미터별로 값을 조정하면서 정확도 끌어올림
            for param in changeParameters:
                # 파라미터 값을 바꿔도 정확도가 안오를때까지 반복
                parameterMove = True  # 파라미터 값을 올릴지 말지 (True면 올린다, False면 내린다)
                parameterMoveStk = 1
                print(param, "의 값 : ", changeParameters[param])
                print("변화 폭 : ", parameterInterval[param] if param in parameterInterval else 0)
                print(param, "의 값 조정 시작")
                while True:
                    # 값을 올리거나 내리고 파라미터 재설정
                    # 값을 올리면서 정확도 올림 올려도 정확도가 안오르면 파라미터를 내려보고 내려도 안오르면 반복 종료
                    if parameterMove == True:
                        if param in parameterInterval:
                            print(param, "의 파라미터 값 증가시킴(", changeParameters[param], "->",
                                  changeParameters[param] + parameterInterval[param], ")")
                            changeParameters[param] += parameterInterval[param]
                        else:
                            print(param, "값 조정 종료")
                            print()
                            break
                    else:
                        if param in parameterInterval:
                            if changeParameters[param] - parameterInterval[param] > 0:
                                print(param, "의 파라미터 값 감소시킴(", changeParameters[param], "->",
                                      changeParameters[param] - parameterInterval[param], ")")
                                changeParameters[param] -= parameterInterval[param]
                            else:
                                print("파라미터 값 못 내림", param, "값 조정 종료")
                                print()
                                changeParameters[param] += (2 - parameterMoveStk) * parameterInterval[param]
                                break

                    curModel.set_params(**changeParameters)
                    # 파라미터가 재설정된 모델을 다시 학습시킴
                    tempScore = cv_silhouette_scorer(curModel, x_train)

                    # 실루엣 스코어로 결정
                    if curResult[2] < tempScore:
                        print("정확도 오름 : ", curResult[2], "->", tempScore, "\n")
                        curResult[1] = changeParameters
                        curResult[2] = tempScore
                        parameterMoveStk = 1
                    elif curResult[2] == tempScore and parameterMoveStk > 0:
                        print("정확도 변화 없음 : ", parameterMoveStk, "번 더 시도")
                        parameterMoveStk -= 1
                        continue
                    else:
                        if parameterMove == True:
                            print("정확도 안오름 반대방향으로 진행", "\n")
                            changeParameters[param] -= (2 - parameterMoveStk) * parameterInterval[param]
                            parameterMove = False
                            parameterMoveStk = 1

                        else:
                            print("정확도 안오름", param, "값 조정 종료")
                            print()
                            changeParameters[param] += (2 - parameterMoveStk) * parameterInterval[param]
                            break

            if bestResult[2] < curResult[2]:
                print("전체 파라미터 조정 후 정확도 상승 - 재반복")
                print()
                bestResult = curResult
            else:
                print("전체 파라미터 조정 후 정확도 상승 없음 - 반복 종료 값 반환")
                print()
                break

    # return 값 : bestResult ([modelObjdect, model_param(dict), score])
    bestResult[0] = initModel.set_params(**bestResult[1])
    print(bestResult, "\n\n")
    return bestResult


df = pd.read_csv("vehicle_coupon.csv")

# pd.set_option('display.max_row', 15)
# pd.set_option('display.max_columns', 100)

# 데이터 정보 & 시각화
dataVisualize(df)

scalers = ["StandardScaler", "MinmaxScaler", "RobustScaler"]
cv = 5

# classification & clustering 모델 및 그에 따른 파라미터 리스트
models_params_classifier = {"DecisionTreeClassifier": {"max_depth": [5, 10], "criterion": ["gini", "entropy"],
                                                       "max_features": [None, "sqrt", "log2", 3, 4, 5],
                                                       "max_leaf_nodes": [5, 10],
                                                       "min_samples_leaf": [5, 10], "min_samples_split": [5, 10],
                                                       "random_state": [5]},
                            "LogisticRegression": {"C": [0.1, 0.5, 1], "max_iter": [500, 1000],
                                                   "solver": ["newton-cg", "lbfgs", "saga"], "random_state": [5]},
                            "RandomForestClassifier": {'criterion': ['gini', "entropy"], "max_depth": [5, 10],
                                                       "max_features": [None, "sqrt", "log2", 3, 4, 5],
                                                       "max_leaf_nodes": [5, 10],
                                                       "max_samples": [5, 10, 20], "n_estimators": [5, 10, 20],
                                                       "random_state": [5]}
                            }
models_params_clustering = {"KMeans": {'n_clusters': [2, 6], 'init': ["k-means++", "random"], 'n_init': [10, 20],
                                       'max_iter': [500, 1000]},
                            "DBSCAN": {'eps': [0.1, 0.5], 'min_samples': [5, 15],
                                       'algorithm': ["auto", "kd_tree"]},
                            "SpectralClustering": {"degree": [2, 6], "gamma": [0.5, 1.0],
                                                   'n_clusters': [2, 6]}}

classificationData = preprocessing(df, False)
classification_scaled_cols = classificationData.columns
classification_scaled_cols = classification_scaled_cols.drop("Y")
classificationFunc(classificationData, scalers, classification_scaled_cols, models_params_classifier, cv)

clusterData = preprocessing(df, True)
clusterData=clusterData.sample(n=3000)
cluster_scaled_cols = clusterData.columns
cluster_scaled_cols = cluster_scaled_cols.drop("Y")
clusteringFunc(clusterData, scalers, cluster_scaled_cols, models_params_clustering, cv)