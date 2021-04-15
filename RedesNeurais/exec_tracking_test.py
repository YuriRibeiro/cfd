#%%
import os
import pathlib
import sys
import time
import yegconfigs
import plot_benchmark_results as pbr
import numpy as np
root = yegconfigs.root
sys.path.append(str(root.parent / 'Submodules')) # Adicionar Submodules ao PATH:
from sort_w import sort


class Tracker():
    def __init__(self):
        pass
    
    @staticmethod#4, 2, 0.33
    def SORT(predictions, max_age=1, min_hits=3, iou_threshold=0.3):
        tracker = sort.Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        mot_labels = {}
        time_sort = 0
        keys = list(predictions.keys())
        keys.sort()
        for frame_number in keys:
            bboxes_atual = predictions[frame_number][:]
            # Formatar a lista para alimentar o Sort
            # np.array( [ [x1,y1,x2,y2,score1], [x3,y3,x4,y4,score2], ... ] )
            if len(bboxes_atual) == 0:
                bboxes_atual = np.zeros((0, 5)) # Requerido pelo Algoritmo Sort
            else:
                for idx in range(len(bboxes_atual)):
                    x1, y1, w, h, score = bboxes_atual[idx][1:]
                    x2 = x1 + w
                    y2 = y1 + h
                    bboxes_atual[idx] = [x1, y1, x2, y2, score]
                
                # Numpy array requerido pelo Sort
                bboxes_atual = np.array(bboxes_atual)
            
                # Analisar o frame atual e identificar os bounding boxes id (update SORT)
                ping = time.time()
                track_bbs_ids = tracker.update(bboxes_atual[:,:-1])
                pong = time.time()
                time_sort += pong-ping
                this_frame_ids = track_bbs_ids[:,-1]
                
                # Passar as coordenadas para o padrão: [frame,idx,x,y,w,h]
                newboxes_list = [[0,0,0,0,0,0,0] for _ in range(len(track_bbs_ids))]
                for i, newbox in enumerate(track_bbs_ids):
                    x1,y1,x2,y2,idx = newbox
                    x, y, w, h = x1, y1, abs(x2-x1), abs(y2-y1)
                    newboxes_list[i] = [frame_number, int(idx), x, y, w, h]
                mot_labels[frame_number] = newboxes_list[:]
        return mot_labels, time_sort

class TrackingData():
    def __init__(self, sort_max_age=1, sort_min_hits=3, sort_iou_threshold=0.3):
        # Best models
        self.byv3, self.byv5 = self._best_det_models()

        # Inference Data
        # UAVDT
        self.resdet_uavdt_fopath = root.parent/'Datasets'/'UAVDT'/'UAV-benchmark-MOTD_v1.0'/'RES_DET'
        self.video_det_paths = [self.resdet_uavdt_fopath / x for x in os.listdir(self.resdet_uavdt_fopath) \
                                    if (x in self.byv3) or (x in self.byv5)]

        # SORT Parameters
        self.sort_max_age=sort_max_age
        self.sort_min_hits=sort_min_hits
        self.sort_iou_threshold=sort_iou_threshold

    def _best_det_models(self):
        yv3 = pbr.Plot_YV3_DET_Bench(auto_calc=True)
        byv3 = yv3.select_best_ap_overall_models(export_to_matlab=False)
        yv5 = pbr.Plot_YV5_DET_Bench(auto_calc=True)
        byv5 = yv5.select_best_ap_overall_models(export_to_matlab=False)
        return byv3, byv5

    def _track_with_SORT(self, predictions):
        return Tracker.SORT(predictions,  self.sort_max_age, self.sort_min_hits, self.sort_iou_threshold)

    def _parse_labels(self, file_path):
        to_float = lambda x: [float(xi) for xi in x]
        with open(file_path) as f:
            data = f.readlines()
        temp = {}
        for line in data:
            frame, _, x, y, w, h, score, _, _ = line.split(',')
            frame, (x, y, w, h, score) = int(frame), to_float([x, y, w, h, score])
            if not frame in temp:
                temp[frame] = [[frame, x, y, w, h, score]]
            else:
                temp[frame].append([frame, x, y, w, h, score])
        return temp
    
    def track_to_bench(self, tracker_name='SORT'):
        '''
            Pega os melhores modelos das redes yv5 e yv3, faz o tracking das detecções obtidas por eles,
            e joga os resultados na pasta correta para que seja possível executar o tracking benchmark _
            com o matlab.

            Os resultados de tempo, em fps, são gravados na pasta RedesNeurais/logs.
        '''
        deltat = 0
        frames = 0

        # Pick best det models
        for video_fopath in self.video_det_paths:
            print(f'[INFO] Analisando {video_fopath}... ')            
            if tracker_name == 'SORT':
                tracker = self._track_with_SORT
                params = [self.sort_max_age, self.sort_min_hits, self.sort_iou_threshold]
                params = [str(i) for i in params]
                sufixo = '-'.join(params)
            else:
                raise Exception(f'Tracker {tracker_name} não encontrado.')            
            
            this_video_deltat = 0
            this_video_frames = 0
            seq_dets_file = video_fopath.glob('*.txt')
            for det_file in seq_dets_file:
                first_line = True
                temp = self._parse_labels(det_file)
                frames += len(temp)
                this_video_frames += len(temp)
                tr_data, time_sort = tracker(temp)
                deltat += time_sort
                this_video_deltat += time_sort
                output_fopath = self.resdet_uavdt_fopath.parent / 'RES_MOT' /video_fopath.stem / f'{tracker_name}-{sufixo}'
                output_fopath.mkdir(parents=True, exist_ok=True)
                output_fipath = output_fopath / f'{det_file.stem}.txt'
                output_fipath_speed = output_fopath.parent / f'speed-{tracker_name}-{sufixo}.txt'
                with open(output_fipath, 'w') as f:
                    if len(tr_data) == 1:
                        f.write('') #video without det
                        break
                    else:
                        for frame_data in tr_data.values(): #tr_data is 1-indexed
                            for data in frame_data:
                                data = data + [1, -1, -1, -1]
                                data = ','.join([str(x) for x in data])
                                if first_line:
                                    f.write(data)
                                    first_line= False
                                else:
                                    f.write('\n'+data)
            with open(output_fipath_speed, 'a') as f:
                f.write(f'Tracker: {tracker_name}; Exp: {video_fopath.stem}; Time: {this_video_deltat} (s). '+\
                        f'Total Frames: {this_video_frames}; FPS: {this_video_frames/this_video_deltat:.6f}\n')
        
        with open(self.resdet_uavdt_fopath.parent / 'RES_MOT' / f'mean-speed-{tracker_name}-{sufixo}', 'a') as f:
            f.write(f'FINAL RESULT: Tracker: {tracker_name}; Total Time: {deltat} (s). Total Frames: {frames}. Mean FPS: {frames/deltat:.6f} (fps)\n')

def rotina_yuri_tracking_bench():
    max_age = [1, 15, 30]
    min_hits = [3, 6, 9]
    iou = [0.3, 0.5, 0.7]
    sort_params = [(x,y,z) for x in max_age for y in min_hits for z in iou]
    for (x,y,z) in sort_params:
        tr_data = TrackingData(sort_max_age=x, sort_min_hits=y, sort_iou_threshold=z)
        tr_data.track_to_bench()


if __name__ == '__main__':
    #rotina_yuri_tracking_bench()
    pass
