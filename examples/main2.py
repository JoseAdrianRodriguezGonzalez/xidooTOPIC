from xidootopic.xidootopic import XidooTopic
import pandas as pd
df=pd.read_csv("../trip/huatulco.csv")
texts=df["comment"].tolist()
model=XidooTopic.default()

result=model.fit_transform(texts)
print(result)
