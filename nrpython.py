#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np

constants = [44483985.77, 254.2, 11.51]

print constants

def f(deflection):
    Displacement = deflection

    CosPhi = 254.0 / pow(pow(254.0, 2.0) + pow(2.54 - Displacement, 2.0), .5)
    SinPhi = (2.54 - Displacement) / pow(pow(254.0, 2.0) + pow(2.54 - Displacement, 2.0), .5)

    Rigidity = constants[0]
    OriginalLength = constants[1]
    kSpring = constants[2]

    return .5 * Rigidity * pow(CosPhi, 2.0) *  pow(Displacement*pow(OriginalLength, -1.0), 2.0)* (Displacement*pow(OriginalLength, -1.0)*pow(CosPhi, 2.0) - 3.0*SinPhi) + kSpring*Displacement + (Rigidity*Displacement*pow(OriginalLength, -1.0))*pow(SinPhi, 2.0)

def j(deflection):
    Displacement = deflection;

    CosPhi = 254.0 / pow(pow(254.0, 2.0) + pow(2.54 - Displacement, 2.0), .5)
    SinPhi = (2.54 - Displacement) / pow(pow(254.0, 2.0) + pow(2.54 - Displacement, 2.0), .5)



    Rigidity = constants[0]
    OriginalLength = constants[1]
    kSpring = constants[2]

    return 1.5*Rigidity*pow(CosPhi,2.0)* (Displacement*pow(OriginalLength, -1.0)*pow(CosPhi, 2.0) -2.0*SinPhi)* Displacement*pow(OriginalLength, -2.0) + kSpring + Rigidity*pow(SinPhi,2.0)*pow(OriginalLength, -1.0) 

D = np.arange(0.0,5.0, .01 )
plt.plot(D, j(D))

plt.plot(D, f(D))
plt.show()

raw_input("Please type something ")
