"""
ECE227 Final Project SP21 by Samuel Cowin and Fatemeh Asgarinejad

Reproduction of the SEIR methodology proposed in this paper:
    "Stability analysis of SEIR model related to efficiency of vaccines for COVID-19 situation" 
    by Phitchayapak Wintachaia and Kiattisak Prathom
"""


from LSTM_utils import LSTM_after_start, vaccination_data
from SEIR_utils import SEIR, plot_model_and_predict

import numpy as np
import warnings
warnings.filterwarnings('ignore')


# Retrieve vaccine data
state = 4
vaccination = vaccination_data()
vaccination_state = list(vaccination.iloc[state])[2:-1]
population = list(vaccination['Population'])
population_state = population[state]

# Fit SEIR with vaccination rates
model = SEIR()
sol = model.fit(days=len(vaccination_state)-1, data=vaccination_state, pop=population_state)
plot_model_and_predict(sol.y)

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
print("Average error is {} percent".format(error))

# Compute error based on population
w = population_state*error
weighted_error = w/sum(population)
print("Weighted error (based on population) is {} percent".format(weighted_error))
