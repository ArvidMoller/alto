# import datetime module
import datetime as dt

end_date = dt.datetime.now()

print(end_date)

end_date = (end_date + dt.timedelta(minutes=(end_date.minute // 15 * 15) - end_date.minute)).isoformat()[:16]
end_date = dt.datetime.fromisoformat(end_date)

print(end_date)