from sklearn import  svm, metrics
from sklearn.model_selection import train_test_split

# ワインデータをファイルを開いて読み込む．
wine_csv=[]
with open ("winequality-white2.csv","r", encoding="utf-8") as fp:
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
    if(pre-1) <=answer <= (pre+1):
        ok +=1
print("ans=",ok, "/",total, "=",ok/total)



# #結果を表示する
# ac_score=metrics.accuracy_score(label_test,predict)
# cl_report=metrics.classification_report(label_test,predict)
# print("ans=",ac_score)
# print("report=\n",cl_report)
