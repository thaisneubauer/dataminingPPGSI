import mushrooms_data
import pandas as pd
from apyori import apriori

data = mushrooms_data.get_data().values.tolist()

results = list(apriori(data))
print(results)

filtered = [result for result in results if result.support >0.5 and result.ordered_statistics[0].confidence >0.7]

print(len(results),len(filtered))