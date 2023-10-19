from sklearn import cross_validation, svm, metricds

wine_csv=[]
with open ("winequality-white.csv"),"r", encoding="utf-8") as fp:
    no=0
    for line in fp:
        line=line.strip()
        cols=line.splot(";")
        wine_csv.append(cols)
        
wine_csv=wine_csv[1:]

label=[]
data=[]
for cols in wine_csv:
    cols=list(map(lamda n : float(n),cols))
    grade=int(cols[11])
    if grade==9: grade=8
    if grade<4: grade=5
    labels.append(grade)
    data.append(cols[0:11])
    