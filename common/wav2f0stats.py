#!/usr/bin/python3
# coding: utf-8

# # Projeto SPIRA
# ## Pós-processamento dos sinais de áudio
# ### Segmentação de trechos de elocução e estatísticas relacionadas a F0
# #### Marcelo Queiroz - Reunião do projeto SPIRA em 08/10/2020
#
# USO: wav2f0stats arquivo.wav limiar_dB

import sys
import os
import math as m
import numpy as np
from scipy import stats
import scipy.io.wavfile as wavfile
import soundfile as sf
import librosa
NOISE_THRESHOLD = float(sys.argv[2])


# abre sinal gravado em arquivo

#filename = openfiledialog.value
#rate, x = opusfile_read('/opt/spira/dados/pacientes/audio/202020/'+filename)
#rate, x = wavfile.read('/home/mqz/research/spira/f0stats/'+filename)
rate, x = wavfile.read(str(sys.argv[1]))
print(str(sys.argv[1]),end=',')


# pré-processamentos: elimina dc e ajusta faixa de amplitudes a [-1,+1]
# (obs: isso não é uma normalização, mas é necessário para permitir
#       o uso da opção normalize=False no widget ipd.Audio)

sample_depth = 8*x[0].itemsize # número de bits por amostra
x = x[:]-np.mean(x) # elimina dc
x = x/(2**(sample_depth-1)) # ajusta amplitudes para [-1,+1]


# calcula a energia média (em dB) do sinal sobre janelas deslizantes
# devolve sinal edB e seu valor mínimo (noise floor)
def window_pow(sig, window_size=4096):
    sig2 = np.power(sig,2)
    window = np.ones(window_size)/float(window_size)
    edB = 10*np.log10(np.convolve(sig2, window))[window_size-1:]
    # corrige aberrações nas extremidades (trechos artificialmente silenciosos)
    imin = int(0.5*rate) # isola 500ms iniciais e finais 
    edBmin = min(edB[imin:-imin])
    edB = np.maximum(edB,edBmin)
    return edB, edBmin

def window_rms(sig, window_size=4096):
    sig2 = np.power(sig,2)
    window = np.ones(window_size)/float(window_size)
    rms = np.sqrt(np.convolve(sig2, window))[window_size-1:]
    return rms


# calcula envoltória de energia em dB do sinal
edB, edBmin = window_pow(x)

# Filtro da mediana 1D para vetores booleanos, sobre janelas de 2*N+1 elementos
def boolean_majority_filter(sig_in,N):
    # cria uma cópia do vetor de entrada
    sig_out = sig_in.copy()
    # coloca N valores True de sentinela antes e depois do vetor sig_in
    # (True força maior chance das bordas serem consideradas ruído, o que é + comum)
    sig_pad = np.concatenate((np.ones(N),sig_in,np.ones(N+1)))
    # contadores de "votos"
    nTrue = 0
    nFalse = 0
    # inicialização da contagem (corresponderá à contagem da situação em sig[0])
    for i in range(2*N+1):
        if sig_pad[i]:
            nTrue += 1
        else:
            nFalse += 1
    # aplica filtro da maioria nos índices de sig_in/out: a cada índice i, o resultado
    # é o voto da maioria na janela sig_in[i-N:i+N+1] (contendo 2*N+1 elementos)
    # que corresponde aos índices sig_pad[i:i+2*N+1] no vetor com sentinelas
    for i in range(len(sig_out)):
        sig_out[i] = nTrue>nFalse
        # subtrai o voto retirado (primeiro da janela deslizante atual)
        # se possível, tira um voto do sinal já *filtrado*
        # (aproveita estabilidade à esquerda)
        if i>=N:
            nout = sig_out[i-N]
        else: # se não for possível, tira um "True" do vetor com sentinelas
            nout = sig_pad[i]
        if nout:
            nTrue -=1
        else:
            nFalse -= 1
        # inclui o voto novo, que é o último da janela deslizante
        # referente ao próximo índice (i+1) em sig
        if sig_pad[i+1+2*N]:
            nTrue += 1
        else:
            nFalse += 1
    return sig_out


# seleção de trechos do sinal sig contendo ruído, devolve sinal booleano
def noise_sel(sig,edB,edBmin,noise_threshold=NOISE_THRESHOLD):
    # seleciona frames com rms próxima do nível mínimo
    inoise_pre = edB<edBmin+noise_threshold
    # aplica filtro da mediana (voto de maioria) para eliminar
    # trechos menores do que 0.2s
    inoise = boolean_majority_filter(inoise_pre,int(0.1*rate))
    return inoise, inoise_pre    


# aplica seleção de frames ao sinal de entrada
xnoise, xnoisep = noise_sel(x,edB,edBmin)


# recorta trechos do sinal identificados como ruído
xx = x.copy()
n = 1
while n<len(x)-1:
    if (xnoise[n] and not xnoise[n-1]) or (xnoise[n-1] and not xnoise[n]):
        for i in range(-100,101):
            if (n+i) in range(len(xx)): xx[n+i] = xx[n+i]*abs(i)/100
        n = n+99
    n = n+1
noise = xx[xnoise]
# recorta trechos de áudio do sinal identificados como locução
xloc = np.logical_not(xnoise)
loc = xx[xloc]


# ## Prova dos 9: trechos marcados como ruído ou elocução


# def printstats():
#     print(f"Duração dos cortes: {noisepercent:.2f}%")
#     print(f"Duração da elocução: {100-noisepercent:.2f}%")
#     print(f"Número de trechos de ruído: {nnoise}")
#     print("Duração dos trechos de ruído:")
#     print(f"\t median: {np.median(noisedur)/rate:2f} seg")
#     print(f"\t mean: {np.mean(noisedur)/rate:2f} seg")
#     print(f"\t std: {np.std(noisedur)/rate:2f} seg")
#     print(f"\t min: {np.min(noisedur)/rate:2f} seg")
#     print(f"\t max: {np.max(noisedur)/rate:2f} seg")
#     print(f"Número de trechos de elocução: {nloc}")
#     print("Duração dos trechos de elocução:")
#     print(f"\t median: {np.median(locdur)/rate:2f} seg")
#     print(f"\t mean: {np.mean(locdur)/rate:2f} seg")
#     print(f"\t std: {np.std(locdur)/rate:2f} seg")
#     print(f"\t min: {np.min(locdur)/rate:2f} seg")
#     print(f"\t max: {np.max(locdur)/rate:2f} seg")


# ## Extração de F0 com YIN

# The standard range is 75–600 Hertz (https://www.fon.hum.uva.nl/praat/manual/Voice.html)
#f0 = librosa.yin(loc,fmin=75,fmax=600,sr=rate)
#plt.plot(f0);plt.title("Curva de F0 instantânea");plt.show()
# Small data decidiu em 5/11/2020 usar 50-600
(f0, pf0, ppf0) = librosa.pyin(loc,sr=rate,fmin=50,fmax=600)

# window_size = 5
# window = np.ones(window_size)/float(window_size)
# f0smooth = np.convolve(f0, window)[window_size-1:]


# phase = 0*loc;
# for n in range(1,len(loc)):
#     f0index = librosa.core.samples_to_frames(n)
#     phase[n] = (phase[n-1]+2*m.pi*f0smooth[f0index]/rate)%(2*m.pi)
# osc = np.sin(phase)*window_rms(loc,2048)

# f0stable = f0.copy()
# nmax = 5
# for n in range(nmax,len(f0)-nmax):
#     maxinterval = 1
#     for i in range(-nmax,nmax+1):
#         maxinterval = max(maxinterval,f0[n]/f0[n+i],f0[n+i]/f0[n])
#     if maxinterval>1.12:
#         f0stable[n] = 0;
#     if  n>2*nmax and stats.mode(f0stable[n-2*nmax:n])==0:
#         f0stable[n-nmax] = 0;
# if f0stable[nmax]==0: f0stable[:nmax]=0
# if f0stable[-nmax]==0: f0stable[-max:]=0
# f0final = f0stable[f0stable>0]
f0final=f0[~np.isnan(f0)]


#print("Median pitch:",np.median(f0final))
#print("Mean pitch:",np.mean(f0final))
#print("Standard deviation:",np.std(f0final))
#print("Minimum pitch:",np.min(f0final))
#print("Maximum pitch:",np.max(f0final))
print(np.median(f0final),end=',')
print(np.mean(f0final),end=',')
print(np.std(f0final),end=',')
print(np.min(f0final),end=',')
print(np.max(f0final))