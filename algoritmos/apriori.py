import mushrooms_data
import votes_data
import pandas as pd
from apyori import apriori

#data = mushrooms_data.get_data().values.tolist()
data = votes_data.get_data().values.tolist()

answer = list(apriori(data, min_support=0.1, min_confidence=0.1))

results = []
for relation in answer:
    items = str(relation[0]).split('{')[1].split('}')[0]
    support = relation[1]
    confidence = relation[2][0][2]
    lift = relation[2][0][3]
    results.append({'items': items, 'support': support, 'confidence':confidence, 'lift': lift})
results_df = pd.DataFrame(results, columns={'items','support','confidence','lift'})
#results_df.to_csv('apriori_results_mushroms.csv', index=False)
results_df.to_csv('apriori_results_house-votes.csv', index=False)