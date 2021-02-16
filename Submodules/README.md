# Pastas de Trabalho:
Para facilitar, pelo menos inicialmente, eu atualizo os submodules e, em seguida, eu mesclo com as "pastas de trabalho" (aquelas que eu realmente utilizo para fazer os experimentos). A intenção é proteger os arquivos quanto a mudanças acidentais e possíveis "bagunças" que possam ser introduzidas pelo git submodule update.

Os "git submodules" aqui servem de referência para a versão que estou utilizando atualmente.

As pastas de trabalho são precedidas por _yuri

# Fluxo de Atualização:
Fluxo para atualizar o Submodule e "mesclar" as atualiações na minha pasta de trabalho:
```bash    
    cd [repo root folder]
    
    git submodule status
    
    cd [inside submodule dir]
    
    git status
    
    git checkout master
    
    git pull

    diff -qr [submodule dir] [destination dir] 
    	
    cp -pur [submodule dir] [destination dir] 
	
    git status

    git add --all
	
    git commit -m "Update SubmoduleName Working Dir."
	
    git push origin
```

# Exclusão de Submodules

Para remover um Submodule, executar os seguintes passos:

    Delete the section referring to the submodule from the .gitmodules file
    Stage the changes via git add .gitmodules
    Delete the relevant section of the submodule from .git/config.
    Run git rm --cached path_to_submodule (no trailing slash)
    Run rm -rf .git/modules/path_to_submodule
    Commit the changes with ```git commit -m "Removed submodule "
    Delete the now untracked submodule files rm -rf path_to_submodule


# Manuais de Treinamento
Conferir os backups de manuais completos postados em:
https://yuriribeiro.github.io/DissertacaoMestrado/



