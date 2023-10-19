# user_input = input("Enter something: ")
# print("You entered:", user_input)

# while True :
#     tubo=input("坪数は")
#     if tubo == "w":
#      break
#     tubo=int(tubo)
#     ms = tubo*3
#     s="{0}ツボは{1}平方です".format(tubo,ms)
#     print(s)


# f={'baa' : 22,'sss' : 24,'asaaaaa' : 44}
# list(f.keys())
# sorted(f.keys())

# def innzei(oricem,sales,per):
#     '''印税わっしょい'''
#     rate = per/100
#     ro=int(price*sales*rate)
#     return ro
# price=int(input("定価は"))
# sales=int(input("発行部数は"))
# pe=float(input("印税率は"))

# print("印税は",innzei(price,sales,pe),"円です")
# help(innzei)

# num = [1,2,3]
# i=iter(num)
# print(i)
# print(next(i))

import random
r=random.randint(1,6)
print(r)