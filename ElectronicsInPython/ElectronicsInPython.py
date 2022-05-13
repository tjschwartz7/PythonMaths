
#chapter 3.10 onwards

import sympy as sym
import math
#from scipy.integrate import quad
from sympy import diff, cos, sin, oo, Symbol, integrate, Piecewise
from sympy.integrals import laplace_transform
from sympy.functions import DiracDelta, exp, Heaviside
from sympy.integrals.deltafunctions import deltaintegrate

from scipy import signal
import matplotlib.pyplot as plt



def iterative_qpoint(f, vd, Is, vd0):
    fp = sym.diff(f)
    out = vd;
    while True:
        vd = out
        out = vd - (float(f.subs(vd0, vd).evalf()) / float(fp.subs(vd0,vd).evalf()))
        if(vd == out):
            break

    current = Is*((sym.exp(40*out))-1)
    print(str(out)+"V"+ " and " + str(current) + "A\n")
 
def integrate(x, value, lower, upper):
  print("Integral of: ")
  sym.pprint(x, use_unicode=False)
  ans = sym.integrate(x, value)
  print("Solution: ")
  sym.pprint(sym.integrate(x, value), use_unicode=False)
  out1 = ans.replace(value, upper)
  out2 = ans.replace(value, lower)
  final = out1 - out2
  print("Substitute t and 0:")
  sym.pprint(final, use_unicode=False)



def laplace_t(x, t, s):
    out = laplace_transform(x, t, s)
    sym.pprint(out, use_unicode=False)
    

def deltaIntegration(f, t):
    out = deltaintegrate(f, t)
    print("Delta integration output:")
    sym.pprint(out.rewrite(Piecewise).doit(), use_unicode=False)

def convolve(f, g, t, lower_limit=-oo, upper_limit=oo):
    tau = Symbol('tau', real=True)
    out = sym.integrate(f.subs(t, tau) * g.subs(t, t - tau), (tau, lower_limit, upper_limit))
    #sym.pprint(out, use_unicode = False)
    sym.pprint(out.rewrite(Piecewise).doit(), use_unicode=False)
    print('\n------------------------------\n')


def symCode():
    vd0 = sym.symbols('vd0')
    #iterative_qpoint(10-10**4*10**-13*(sym.exp(40*vd0)-1),1.0, 10**-13, vd0)
    #iterative_qpoint(10-10**4*10**-15*(sym.exp(40*vd0)-1),1.0, 10**-15, vd0)

    eta, eta_prime, lamb, beta, temp, alpha = sym.symbols('eta eta_prime lamb beta temp alpha', real=True)
    t, s = sym.symbols('t s')
    c,w = sym.symbols('c w')
    #integrate(sym.exp(lamb)*sym.exp(-2*t),  lamb, 0, t )
    #integrate(1,  lamb, 0, t )
    #integrate(sym.exp(-c*t),  lamb, 0, t )
    #integrate(lamb,  lamb, 0, t )
    #integrate(sym.exp(-c*t)+sym.cos(w*t), t,0, t)
    #deltaIntegration((exp(-alpha*t)+sym.cos(eta*t))*DiracDelta(t), t)
    #deltaIntegration((exp(-3*t))*diff(DiracDelta(t-3), t), t)
    print("Convolution1:")
    convolve(Heaviside(t+3, 1)-Heaviside(t+2, 1), exp(-1*t)*Heaviside(t, 1), t)
    print("Convolution2:")
    convolve(Heaviside(t+4, 1)-Heaviside(t, 1), 3*exp(-2*t)*Heaviside(t, 1), t)

def sciCode():
    sys = signal.lti([1.0], [1.0,1.0])
    t, y = signal.step(sys)
    plt.plot(t,y)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title("Step response for 1. Order lowpass")
    plt.grid()
    plt.show()

symCode()
#sciCode()

#x = sym.Symbol('x')
#out1 = integrate((25/2)*((x**2)/2), x, 0, 5)
#out2 = integrate((25/2)*((10-x)/2), x, 5, 10)
#sym.pprint(out1, use_unicode = False)
#sym.pprint(out2, use_unicode = False)
