# import datetime module
import datetime

# consider the start date as 2021-february 1 st
start_date = datetime.datetime(2025, 10, 1, 10, 00, 00, 000)

# consider the end date as 2021-march 1 st
end_date = datetime.datetime(2025, 10, 17, 10, 00, 00, 000)

# delta time
delta = datetime.timedelta(minutes=15)

# iterate over range of dates
while (start_date <= end_date):
    print(start_date, end="\n")
    start_date += delta
    


print(start_date.year)

print(f"{start_date.year}-{start_date.month}-{start_date.day}T{start_date.hour}:{start_date.minute}:00.000")