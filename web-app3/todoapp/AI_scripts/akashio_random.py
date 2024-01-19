from sklearn import  svm, metrics
from sklearn.model_selection import train_test_split
import pandas as pd

#今後はこのyosokuという関数に引数を入れれるようにしたい．
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
# import seaborn as sns
import matplotlib.pyplot as plt
def yosoku2():
    # CSVファイルを読み込む
    df = pd.read_csv("todoapp/AI_scripts/CSV/HIU_data_+n.csv")

    # 空白をNaNに変換
    df.replace(' ', pd.NA, inplace=True)

    # NaNを含む行を削除
    df.dropna(inplace=True)

    # Chl.aを3つのクラスに分割
    df['label_class'] = pd.cut(df['Chl.a'], bins=[-np.inf, 1 ,2,5,10,20,np.inf,], labels=[0,1,2,3,4,5])

    # 説明変数として使用する列を選択
    selected_columns = ['Tem', 'DO','Sal', 'nissyaryou']
    # 正解データの列を指定
    label_column = 'label_class'

    # データの前処理
    labels = df[label_column]
    data = df[selected_columns]

    # 訓練データとテストデータに分割
    data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.1, random_state=0)

    # ランダムフォレストのアルゴリズムを利用して学習
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(data_train, label_train)

    # テストデータで予測
    label_pred = clf.predict(data_test)

    # 混同マトリックスを取得
    conf_matrix = confusion_matrix(label_test, label_pred)

    # 分類精度を表示
    accuracy = accuracy_score(label_test, label_pred)
    
    new_data = [[25.0,7.5,35.0,5.0]]
    predicted_label = clf.predict(new_data)
    string0="かなり安全"
    string1="安全"
    string2="注意が必要"
    string3="危険"
    string4="かなり危険"
    
    
    if predicted_label==0:
        return string0
    elif predicted_label==1:
        return string1
    elif predicted_label==2:
        return string2
    elif predicted_label==3:
        return string3
    elif predicted_label==4:
        return string4
    
    
   
    # print(f"Accuracy: {accuracy:.3f}")

    # # 分類レポートを表示
    # report = classification_report(label_test, label_pred)
    # print("Classification Report:")
    # print(report)

    # # 混同マトリックスを表示
    # print("Confusion Matrix:")
    # print(conf_matrix)

    # # 混同マトリックスを可視化
    # labels = df['label_class'].cat.categories
    # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.show()






def yosoku():
    # ワインデータをファイルを開いて読み込む．
    wine_csv=[]



            
    with open ("todoapp/AI_scripts/CSV/number.csv","r", encoding="utf-8") as fp:
        no=0
        for line in fp:
            line=line.strip()
            cols=line.split(";")
            wine_csv.append(cols)
    
    #ひうちなだのデータをよみこませたい（まだできない）
    # with open ("todoapp/AI_scripts/CSV/HIU_data_+n.csv","r", encoding="utf-8") as fp:
    #     no=0
    #     for line in fp:
    #         line=line.strip()
    #         cols=line.split(",")
    #         wine_csv.append(cols)
            
   
            
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



