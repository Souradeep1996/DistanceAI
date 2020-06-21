#import dependencies
import pandas as pd
import datetime

todays_date = datetime.datetime.now().date()
index = pd.date_range(start=todays_date-datetime.timedelta(6), end= todays_date,  freq='D')
columns = ['DIP Gate', 'Main Gate']
df = pd.DataFrame(index=index, columns=columns)
df = df.fillna(0) # with 0s rather than NaNs
df.at['15-06-2020', 'DIP Gate'] = 1047
df.at['15-06-2020', 'Main Gate'] = 358
df.at['16-06-2020', 'DIP Gate'] = 954
df.at['16-06-2020', 'Main Gate'] = 224
df.at['17-06-2020', 'DIP Gate'] = 1032
df.at['17-06-2020', 'Main Gate'] = 296
df.at['18-06-2020', 'DIP Gate'] = 864
df.at['18-06-2020', 'Main Gate'] = 213
df.at['19-06-2020', 'DIP Gate'] = 996
df.at['19-06-2020', 'Main Gate'] = 226
df.at['20-06-2020', 'DIP Gate'] = 254
df.at['20-06-2020', 'Main Gate'] = 367
df.at['21-06-2020', 'Main Gate'] = 24
df.at['21-06-2020', 'DIP Gate'] = 12
