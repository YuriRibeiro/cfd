# -*- coding: utf-8 -*-
"""
Spyder Editor

Lista de todos os environments conda e pacotes instalados na máquina.
    "conda env export --from-history --name %s"%(env_name)

Este código funciona apenas em Linux, Bash.

Return:
    Arquivo texto com todos os environments e pacotes instalados pelo
    usuário.
"""

import subprocess
from platform import uname
from os import environ
# =============================================================================
# Capturar o nome dos conda environments instalados
# Resultado gravado na variável envs_list.
# =============================================================================
params = {"stdout" : subprocess.PIPE,
          "text" : True,
          "check": True}

username = environ['USER']
hostname = uname().node
sysinfo = uname()

cmd1 = "conda info --envs".split(" ")
envs = subprocess.run(cmd1, **params).stdout.split("\n")[2:-2]

envs_list = [None]*len(envs)
for idx, env in enumerate(envs):
    #env name should not have spaces, otherwise BUG..
    env_name = env.split(" ")[0]
    envs_list[idx] = env_name

#print(envs_list)

# =============================================================================
# Para cada environment, carregar os pacotes instalados --from-history
# =============================================================================
packages = [None]*len(envs)
for idx, env_name in enumerate(envs_list):
    cmd2 = "conda env export --from-history --name %s"%(env_name)
    packages[idx] = subprocess.run(cmd2.split(" "), **params).stdout

#for packs in packages:
 #   print(packs)

# =============================================================================
# Salvar em um arquivo de texto "conda_envs_user_hostname.txt"
# =============================================================================
with open("./.infos/conda_envs_%s@%s.txt"%(username, hostname), "w") as file:
    file.write("Computer Info:\n\n" + 
               str(sysinfo)[13:-1]+
               '\n\nConda Environments:\n\n')
    for packs in packages:
        file.write(packs)
        file.flush()
        
