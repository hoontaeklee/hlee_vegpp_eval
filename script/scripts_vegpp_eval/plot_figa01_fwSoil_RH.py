'''
plot the response curve of gsi f(wSoil) with varying parameters
'''

#%% 
#%% 
import numpy as np
import xarray as xr
from csaps import csaps
import matplotlib.pyplot as plt

'''
wSoil_top = 100 * wSoil / wSoil_sat

%--> first half of the response curve
W2p1    =   1 ./ (1 + exp(A .* (-10))) ./ (1 + exp(A .* (- 10)));
W2C1    =   1 ./ W2p1;
W21     =   W2C1 ./ (1 + exp(A .* (WOPT - 10 - wSoil_top))) ./ ...
            (1 + exp(A .* (- WOPT - 10 + wSoil_top)));

%--> second half of the response curve
W2p2    =   1 ./ (1 + exp(B .* (-10))) ./ (1 + exp(B .* (- 10)));
W2C2    =   1 ./ W2p2;
T22     =   W2C2 ./ (1 + exp(B .* (WOPT - 10 - wSoil_top))) ./ ...
            (1 + exp(B .* (- WOPT - 10 + wSoil_top)));

%--> combine the response curves
v       =   wSoil_top >= WOPT;
T2      =   W21;
T2(v)   =   T22(v);
'''

def fwsoil_segment(wSoil, WOPT, par):
    W2p1 = 1 / (1 + np.exp(par * (-10))) / (1 + np.exp(par * (-10)))
    W2C1 = 1 / W2p1
    W21  = W2C1 / (1 + np.exp(par * (WOPT - 10 - wSoil))) / (1 + np.exp(par * (-WOPT -10 + wSoil)))

    return W21

def fwsoil_gsi(wSoil, WOPT=0.5, A=0.2, B=0.2):
    W21  = fwsoil_segment(wSoil=wSoil, WOPT=WOPT, par=A)
    T22  = fwsoil_segment(wSoil=wSoil, WOPT=WOPT, par=B)

    return np.where(wSoil>=WOPT, T22, W21)

wSoil = np.linspace(0, 100, 100)

ar_A = np.linspace(0.01, 0.1, 3)
ar_B = np.linspace(0.01, 1.0, 3)
ar_fwSoil = np.arange(100) * np.nan
ncomb = ar_A.size * ar_B.size
dict_fwSoil = {k:{'WOPT':0, 'A':0, 'B':0, 'fwSoil':[]} for k in np.arange(ncomb+1)}

for i in range(ncomb):
    WOPT = 87.83
    i_a = i//3
    i_b = i%3
    fwSoil = fwsoil_gsi(wSoil=wSoil, WOPT=87.83, A=ar_A[i_a], B=ar_B[i_b])
    dict_fwSoil[i]['WOPT'] = WOPT
    dict_fwSoil[i]['A'] = ar_A[i_a]
    dict_fwSoil[i]['B'] = ar_B[i_b]
    dict_fwSoil[i]['fwSoil'] = fwSoil

fwSoil_opt = fwsoil_gsi(wSoil=wSoil, WOPT=87.83, A=0.05, B=0.79)
dict_fwSoil[ncomb]['WOPT'] = 87.83
dict_fwSoil[ncomb]['A'] = 0.05
dict_fwSoil[ncomb]['B'] = 0.79
dict_fwSoil[ncomb]['fwSoil'] = fwSoil_opt

#%% plot
plt.style.use('seaborn-whitegrid')
fig, axes = plt.subplots(1, 1, figsize=(4.8*1.5, 2.7*1.5))

for i in range(ncomb):
    _a = dict_fwSoil[i]['A']
    _b = dict_fwSoil[i]['B']
    axes.plot(wSoil, dict_fwSoil[i]['fwSoil'], alpha=0.3, color='grey', label=f'A={_a.round(2)}, B={_b.round(2)}')
axes.axvline(x=87.83, linestyle='dashed', linewidth=1.0, color='black')
axes.set_xlabel('wSoil (%)')
axes.set_ylabel('f(wSoil) (-)')
axes.annotate('Wopt', (0.81, 1.02), xycoords='axes fraction')
axes.annotate('A=0.01', (0.53, 0.88), xycoords='axes fraction')
axes.annotate('A=0.06', (0.53, 0.43), xycoords='axes fraction')
axes.annotate('A=0.10', (0.53, 0.13), xycoords='axes fraction')
axes.annotate('B=0.01', (0.93, 0.98), xycoords='axes fraction')
axes.annotate('B=0.50', (0.98, 0.28), xycoords='axes fraction')
axes.annotate('B=1.0', (0.88, 0.09), xycoords='axes fraction')

axes.plot(wSoil, dict_fwSoil[ncomb]['fwSoil'], alpha=0.9, linewidth=1, color='red', label=f'optimized')
axes.annotate('Optimized', (0.28, 0.38), xycoords='axes fraction', color='red')
# fig.legend()

fig.savefig(
    '/Net/Groups/BGI/people/hlee/hlee_vegpp_eval/figures/figa01_fwSoil_RH.png',
    dpi=600,
    bbox_inches='tight',
    facecolor='w',
    transparent=False
)