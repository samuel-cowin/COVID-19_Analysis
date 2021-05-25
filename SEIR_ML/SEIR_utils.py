"""
ECE227 Final Project SP21 by Samuel Cowin and Fatemeh Asgarinejad

Reproduction of the SEIR methodology proposed in this paper:
    "Stability analysis of SEIR model related to efficiency of vaccines for COVID-19 situation" 
    by Phitchayapak Wintachaia and Kiattisak Prathom
"""


from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class SEIR:

    def __init__(self):
        """
        Initializing parameters from paper for fixed rates regarding state transitions and vaccine effectiveness
        """

        self.alpha = 6
        self.beta = 16
        self.gamma = 0.001
        self.p_s = 0.35
        self.p_e = 0.1
        self.p_i = 0.0
        self.b_0 = 0.0
        self.d_0 = 0.0
        self.d_1 = 0.0
        self.d_2 = 0.0


    def dS_dt(self, v, S, I):
        """
        Modeling susceptible characteristics as:
        dS_dt = b_0 - (v*p_s+d_0)*S - beta*(1-v*p_s)*S*I

        Where parameters are as follows:
            b_0: birth rate of population
            v: vaccination rate
            p_s: vaccination effectiveness for S
            d_0: death rate without COVID-19
            S: fraction of susceptible cases
            beta: transmission rate of COVID-19
            I: fraction of infectious cases
        """

        return self.b_0 - (v*self.p_s+self.d_0)*S - self.beta*(1-v*self.p_s)*S*I


    def dE_dt(self, v, S, E, I):
        """
        Modeling exposed characteristics as:
        dE_dt = beta*(1-v*p_s)*S*I - (d_1+alpha+(1-alpha)*v*p_e)*E

        Where parameters are as follows:
            beta: transmission rate of COVID-19
            v: vaccination rate
            p_s: vaccination effectiveness for S
            S: fraction of susceptible cases
            I: fraction of infectious cases
            d_1: death rate of exposed population plus d_0
            alpha: rate change from E to I
            p_e: vaccination effectiveness for E
            E: fraction of exposed cases
        """

        return self.beta*(1-v*self.p_s)*S*I - (self.d_1+self.alpha+(1-self.alpha)*v*self.p_e)*E


    def dI_dt(self, v, E, I):
        """
        Modeling infected characteristics as:
        dI_dt = alpha*E - (d_2+gamma+(1-gamma)*v*p_i)*I

        Where parameters are as follows:
            alpha: rate change from E to I
            E: fraction of exposed cases
            d_2: death rate of infected population plus d_0
            gamma: rate change from I to R
            v: vaccination rate
            p_i: vaccination effectiveness for I
            I: fraction of infectious cases
        """

        return self.alpha*E - (self.d_2+self.gamma+(1-self.gamma)*v*self.p_i)*I


    def dR_dt(self, v, S, E, I, R):
        """
        Modeling infected characteristics as:
        dR_dt = v*p_s_S + v*p_e*(1-alpha)*E + (gamma+(1-gamma)*v*p_i)*I - d_0*R

        Where parameters are as follows:
            v: vaccination rate
            p_s: vaccination effectiveness for S
            S: fraction of susceptible cases
            p_e: vaccination effectiveness for E
            alpha: rate change from E to I
            E: fraction of exposed cases
            gamma: rate change from I to R
            p_i: vaccination effectiveness for I
            I: fraction of infectious cases
            d_0: death rate without COVID-19
            R: fraction of recovered cases
        """

        return v*self.p_s*S + v*self.p_e*(1-self.alpha)*E + (self.gamma+(1-self.gamma)*v*self.p_i)*I - self.d_0*R


    def get_v_rate(self, data, pop, time):
        """
        Change dataset into callable vaccination rate
        """

        if time==0:
            return 0
        else:
            return data[int(time)]/pop


    def SEIR(self, t, state, v_rate, data, pop, call_v=True):
        """
        SEIR model implementation 
            v_rate: vaccination rate as float or callable
            S: susceptible population estimate
            E: exposed population estimate
            I: infectious population estimate
            R: recovered population estimate
        """

        if call_v:
            v = self.get_v_rate(data, pop, t)
        else:
            v = v_rate

        S, E, I, R = state
        
        S_change = S*self.dS_dt(v, S, I)
        E_change = E*self.dE_dt(v, S, E, I)
        I_change = I*self.dI_dt(v, E, I)
        R_change = R*self.dR_dt(v, S, E, I, R)

        return [S_change, E_change, I_change, R_change]


    def fit(self, data, pop, v=0.0, call=True, S_init=0.994, E_init=0.002, I_init=0.002, R_init=0.002, days=5000):
        """
        Method to fit the SEIR model and solve the ODEs for presentation
            S_init: initial percentage of the population that is susceptible
            E_init: initial percentage of the population that is exposed
            I_init: initial percentage of the population that is infected
            R_init: initial percentage of the population that is recovered
            days: simulation duration
        """

        s = S_init
        e = E_init
        i = I_init
        r = R_init
        v_rate = v

        sol = solve_ivp(fun=self.SEIR, t_span=[0, days], y0=[s, e, i, r], args=[v_rate, data, pop, call], t_eval=list(range(days+1)))

        return sol


def plot_model_and_predict(solution, title='SEIR model'):
    """
    Plotting method for the SEIR modeling characteristics
    """

    [sus, exp, inf, rec] = solution
    
    f = plt.figure(figsize=(16,5))
    plt.plot(sus, 'b', label='Susceptible')
    plt.plot(exp, 'y', label='Exposed')
    plt.plot(inf, 'r', label='Infected')
    plt.plot(rec, 'c', label='Recovered/deceased')
    plt.title(title)
    plt.xlabel("Days", fontsize=10)
    plt.ylabel("Fraction of population", fontsize=10)
    plt.ylim([0, 0.5])
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    model = SEIR()
    sol = model.fit()
    plot_model_and_predict(sol.y)