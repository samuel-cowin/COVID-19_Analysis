"""
ECE227 Final Project SP21 by Samuel Cowin and Fatemeh Asgarinejad

SEIR model with vaccination rate predictions using Deep Learning
"""


from LSTM_utils import LSTM_after_start, vaccination_data
from SEIR_utils import SEIR, plot_model_and_predict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


"""
SEIR with set vaccination rates
"""

# Fit SEIR with changing fixed vaccination rates
model1 = SEIR()
sol = model1.fit(data=[], pop=[], v=0.0, days=500, call=False)
plot_model_and_predict(sol.y, title='SEIR - v=0.0')
sol = model1.fit(data=[], pop=[], v=0.1, days=500, call=False)
plot_model_and_predict(sol.y, title='SEIR - v=0.1')
sol = model1.fit(data=[], pop=[], v=1.0, days=500, call=False)
plot_model_and_predict(sol.y, title='SEIR - v=1.0')


"""
Initial SEIR fit with real vaccination data
"""

# Retrieve vaccine data
state = 4
vaccination = vaccination_data()
vaccination_state = list(vaccination.iloc[state])[2:-1]
population = list(vaccination['Population'])
population_state = population[state]

# Retrieve real confirmed cases
confirmed = pd.read_csv('../Dataset/Confirmed.csv', error_bad_lines=False)
confirmed_norm = [c/population_state for c in list(confirmed.iloc[state+1])[1:]]

# Fit SEIR with vaccination rates
model2 = SEIR()
sol = model2.fit(days=len(vaccination_state)-1, data=vaccination_state, pop=population_state)
plot_model_and_predict(sol.y, title='SEIR - known vaccination rates')

[sus, exp, inf, rec] = sol.y
f = plt.figure(figsize=(16,5))
plt.plot(inf, 'b', label='SEIR')
plt.plot(confirmed_norm, 'r', label='Real')
plt.title('Infected - known SEIR vs. Real')
plt.xlabel("Time", fontsize=10)
plt.ylabel("Fraction of population", fontsize=10)
plt.ylim([0, 0.2])
plt.legend(loc='best')
plt.show()


"""
SEIR fit extension for predicted vaccination rates
"""

# Run LSTM on sequencing data for California
predictions_May_18_after_start = []
lstm, shape = LSTM_after_start(vaccination_state, 100)

# Predict on final California Data
x_input = np.array(list(vaccination_state[-4:-1]))
x_input = x_input.reshape(1, shape[1], shape[2])
E = int(lstm.predict(x_input, verbose=0))
T =(vaccination_state[-1])

# Compute error
error = (E-T)/T*100
print("Average error is {} percent for initial prediction".format(error))

# Compute error based on population
w = population_state*error
weighted_error = w/sum(population)
print("Weighted error (based on population) is {} percent".format(weighted_error))

# Extension of prediction
x_pred = []
for i in range(len(confirmed_norm)-len(vaccination_state)):
    x_pred = np.append(x_pred, lstm.predict(x_input, verbose=0))
    x_input = np.append(x_input, lstm.predict(x_input, verbose=0))
    x_input = x_input[1:]
    x_input = x_input.reshape(1, shape[1], shape[2])

# # Fit SEIR with added vaccination rates
new_vaccination_state = np.concatenate((vaccination_state, x_pred))
model3 = SEIR()
sol = model3.fit(days=len(new_vaccination_state)-1, data=new_vaccination_state, pop=population_state)
plot_model_and_predict(sol.y, title='SEIR - predicted vaccination rates')

[sus, exp, inf, rec] = sol.y
f = plt.figure(figsize=(16,5))
plt.plot(inf, 'b', label='SEIR')
plt.plot(confirmed_norm, 'r', label='Real')
plt.title('Infected - predicted SEIR vs. Real')
plt.xlabel("Time", fontsize=10)
plt.ylabel("Fraction of population", fontsize=10)
plt.ylim([0, 0.2])
plt.legend(loc='best')
plt.show()