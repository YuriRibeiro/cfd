# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:08:14 2020

@author: Ribeiro
"""
from time import time
from functools import wraps

def medirTempoExecucao(funcao):
    @wraps(funcao)
    def wraper(*args, **kwargs):
        initial_time = time()
        funcao(*args, **kwargs)
        final_time   = time()
    
        # Formata a mensagem que será mostrada na tela
        print(
            "[{funcao}] Tempo total de execução: {tempo_total}".format(
                funcao=funcao.__name__,
                tempo_total=str(final_time - initial_time)            )
             )
    return wraper

# Example of use
if __name__ == "__main__":
    @medirTempoExecucao
    def main(N):
        for _ in range(N):
            pass

    main(N = int(1E6))
