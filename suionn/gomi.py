from datetime import date, timedelta


current_day = date.today()
yesterday_day = date.today() - timedelta(days=1)


print(current_day) #2022-06-29
print(yesterday_day) #2022-06-28