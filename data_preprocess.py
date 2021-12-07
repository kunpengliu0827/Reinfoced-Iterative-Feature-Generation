import pandas as pd
data = []
with open("boston_data.csv", "r") as f:
    lines = f.readlines()
    for line in lines:
        items = line.split(" ")
        row = []
        for item in items:
            if len(item) != 0:
                row.append(float(item))
        data.append(row)
data = pd.DataFrame(data)
data.to_csv("new_boston_data.csv",index=False)
