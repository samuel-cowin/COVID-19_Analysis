"""
ECE227 Final Project SP21 by Samuel Cowin and Fatemeh Asgarinejad

SEIR model with vaccination rate predictions using Deep Learning
"""


from LSTM_utils import LSTM_after_start, vaccination_data
from SEIR_utils import SEIR, plot_model_and_predict, compute_mse

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
sol1 = model1.fit(data=[], pop=[], v=0.0, days=420, call=False)
# plot_model_and_predict(sol1.y, title='SEIR - v=0.0')
sol2 = model1.fit(data=[], pop=[], v=0.01, days=420, call=False)
# plot_model_and_predict(sol2.y, title='SEIR - v=0.01')
sol3 = model1.fit(data=[], pop=[], v=0.1, days=420, call=False)
# plot_model_and_predict(sol3.y, title='SEIR - v=0.1')


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
confirmed = pd.read_csv('../Dataset/Combined_data/Active_Daily.csv', error_bad_lines=False)
confirmed_norm = [c/population_state for c in list(confirmed.iloc[state+1])[2:]]
print(confirmed.iloc[state+1])

# Fit SEIR with vaccination rates
model2 = SEIR()
sol4 = model2.fit(days=len(vaccination_state)-1, data=vaccination_state, pop=population_state)
# plot_model_and_predict(sol4.y, title='SEIR - known vaccination rates')

[sus, exp, inf, rec] = sol4.y
f = plt.figure(figsize=(16,5))
plt.plot(inf, 'b', label='SEIR')
plt.plot(confirmed_norm, 'r', label='Real')
plt.title('Infected - known SEIR vs. Real')
plt.xlabel("Time", fontsize=10)
plt.ylabel("Fraction of population", fontsize=10)
plt.ylim([0, 0.005])
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
sol5 = model3.fit(days=len(new_vaccination_state)-1, data=new_vaccination_state, pop=population_state)
# plot_model_and_predict(sol5.y, title='SEIR - predicted vaccination rates')

rates = []
R0 = []
for v in range(1, len(new_vaccination_state)):
    rates.append((new_vaccination_state[v]-new_vaccination_state[v-1])/population_state)
    R0.append(model1.get_R0(v=new_vaccination_state[v]))

plt.plot(rates, 'r', label='V rate')
# plt.plot(R0, 'b', label='R_0')
plt.title('Vaccinations rate - predicted', fontsize=30)
plt.xlabel("Time", fontsize=20)
plt.ylabel("Fraction of population", fontsize=20)
# plt.ylim([0, 0.005])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# plt.legend(loc='best')
plt.show()


"""
Final plotting comparison
"""

[sus1, exp1, inf1, rec1] = sol1.y
[sus2, exp2, inf2, rec2] = sol2.y
[sus3, exp3, inf3, rec3] = sol3.y
[sus4, exp4, inf4, rec4] = sol4.y
[sus5, exp5, inf5, rec5] = sol5.y
f = plt.figure(figsize=(16,5))
plt.plot(inf1, 'g', label='SEIR - v=0.0')
plt.plot(inf2, 'c', label='SEIR - v=0.01')
plt.plot(inf3, 'm', label='SEIR - v=0.1')
plt.plot(inf4, 'k', label='SEIR - v real')
plt.plot(inf5, 'b', label='SEIR - v predicted')
plt.plot(confirmed_norm, 'r', label='Real')
plt.title('Infected - predicted SEIR vs. Real', fontsize=30)
plt.xlabel("Time", fontsize=20)
plt.ylabel("Fraction of population", fontsize=20)
plt.ylim([0, 0.005])
plt.legend(loc='best', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()


"""
Final plotting comparison - shorten timeline
"""

start = 215
[sus1, exp1, inf1, rec1] = sol1.y
[sus2, exp2, inf2, rec2] = sol2.y
[sus3, exp3, inf3, rec3] = sol3.y
[sus4, exp4, inf4, rec4] = sol4.y
[sus5, exp5, inf5, rec5] = sol5.y
f = plt.figure(figsize=(16,5))
plt.plot(inf1[:start], 'g', label='SEIR - v=0.0')
plt.plot(inf2[:start], 'c', label='SEIR - v=0.01')
plt.plot(inf3[:start], 'm', label='SEIR - v=0.1')
plt.plot(inf4[:start], 'k', label='SEIR - v real')
plt.plot(inf5[:start], 'b', label='SEIR - v predicted')
plt.plot(confirmed_norm[start:], 'r', label='Real')
plt.title('Infected - predicted SEIR vs. Real', fontsize=30)
plt.xlabel("Time", fontsize=20)
plt.ylabel("Fraction of population", fontsize=20)
plt.ylim([0, 0.005])
plt.legend(loc='best', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()


"""
Compute MSE loss between real and predicted 
"""

mse = compute_mse(confirmed_norm, inf5)