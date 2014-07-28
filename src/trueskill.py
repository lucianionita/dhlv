import trueskill as ts
import numpy as np


ratings = {}

def get_rating(horse):
    if not ratings.has_key(horse):
        ratings[horse] = ts.Rating()
    return ratings[horse]


races = data['Season-#'].unique()
races.sort()

for race in races:
    print "Race: ", race
    d = data[data['Season-#']==race]
    rdict = {}
    horses = d['Horse'].values
    places = d['FP'].values-1
    rtgs = [(get_rating(horse),) for horse in horses]
    
            
    

g.name = "OutsideLaneNo"
rdata = pd.DataFrame(g)
data = pd.merge(data, rdata, how='outer', left_on = 'Season-#', right_index=True)
