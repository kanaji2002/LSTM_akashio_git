from sklearn import  svm, metrics
from sklearn.model_selection import train_test_split
import pandas as pd

#今後はこのyosokuという関数に引数を入れれるようにしたい．
def yosoku():
    # ワインデータをファイルを開いて読み込む．
    wine_csv=[]


    # with open ("web-app2/todoapp/AI_scripts/CSV/number.csv","r", encoding="utf-8") as fp:
    #     no=0
    #     for line in fp:
    #         line=line.strip()
    #         cols=line.split(";")
    #         wine_csv.append(cols)
            
    with open ("todoapp/AI_scripts/CSV/number.csv","r", encoding="utf-8") as fp:
        no=0
        for line in fp:
            line=line.strip()
            cols=line.split(";")
            wine_csv.append(cols)
            
   
            
    #1行目はヘッダ行なので削除
    wine_csv=wine_csv[1:]


    #CSVの各データを数値に変換
    labels=[]
    data=[]
    for cols in wine_csv:
        cols=list(map(lambda n : float(n),cols))
        grade=int(cols[11])
        if grade==9: grade=8
        if grade<4: grade=5
        labels.append(grade)
        data.append(cols[0:11])
        
    #訓練ようと，テスト用にデータを分ける．
    data_train, data_test,label_train,label_test = \
        train_test_split(data,labels)
        
    #SVMのアルゴリズムを利用して学習
    # clf=svm.SVC()
    # clf.fit(data_train, label_train)

    #ランダムフォレストのアルゴリズムを利用して学習
    from sklearn.ensemble import RandomForestClassifier
    clf=RandomForestClassifier()
    clf.fit(data_train, label_train)

    #予測してみる
    predict=clf.predict(data_test)
    total=ok=0
    for idx, pre in enumerate(predict):
        # pre = predict[idx] #予測したラベル
        answer=label_test[idx] #正解ラベル
        total +=1
    #ほぼ正解なら，正解とみなす．
        #if(pre-1) <=answer <= (pre+1):
        if answer == pre:
            ok +=1
    #print("ans=",ok, "/",total, "=",ok/total)

    


    # 新しいデータで予測してみる
    new_data = [[3, 4, 8, 9, 2, 3, 4, 5, 6, 7, 1]]
    predicted_grade = clf.predict(new_data)
    # print("Predicted Grade:", predicted_grade)

    # 結果を表示する
    ac_score = metrics.accuracy_score(label_test, predict)
    cl_report = metrics.classification_report(label_test, predict)
    # print("Accuracy Score:", ac_score)
    # print("Classification Report:\n", cl_report)

    return predicted_grade

# 予想結果をyosoku.htmlファイルに表示させる関数



