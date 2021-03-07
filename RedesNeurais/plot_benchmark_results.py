import yegconfigs
import pathlib
import pandas as pd

class Populate_Benchmark(yegconfigs.YV5_CONFIGS):
    """
    Esta classe implementa métodos para mover os arquivos YOLOv*_UAVDT_*/YOLO
    """  
    def __init__(self): pass
    
    @staticmethod
    def gather_paths():
        det_paths_dict = Populate_Benchmark.get_det_paths()        
        resolutions = [f'{q}x{q}' for q in range(256, 1536+128, 128)]
        dic = {}
        for name, path in det_paths_dict.items():
            det_txt_folder_paths = [x for x in path.glob('[0-9]*x[0-9]*') if x.is_dir()]
            if det_txt_folder_paths == []: print(f'Name: {name}. Path: {path}. Detecções não encontradas.')
            list_ = [None]*len(resolutions)
            for folderpath in det_txt_folder_paths:
                res = folderpath.stem
                if not res in resolutions: print("Problemas na resolução.{res} não encontrada.")
                list_[resolutions.index(res)] = folderpath
            dic[name] = list_ 

        df = pd.DataFrame(dic, columns=resolutions)
        print(df)
        return df

class YV5_DET(Populate_Benchmark):
    pass

a = print(Populate_Benchmark().gather_paths())