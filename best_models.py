import numpy as np
import matplotlib.pyplot as plt


#Open list with winning models in every big epoch
best_models_file = open("best_models_list.txt", "r") 

data = best_models_file.read()

best_models_list = data.split("\n") 

 
best_models_file.close()

best_models_list.pop()

#Dictionary for model names and mapping for analysis
dict_for_map={'fast_hunter':4,'normal_hunter':3, 'not_so_slow_hunter':2, 'slow_hunter':1}

mapped_list=[*map(dict_for_map.get,best_models_list)]

mapped_list=np.array(mapped_list)
print(mapped_list)

#Dynamic of winning models
plt.plot(mapped_list, label='Fast: 4, Normal: 3, Not so slow: 2, Slow: 1')
plt.legend()
plt.show()