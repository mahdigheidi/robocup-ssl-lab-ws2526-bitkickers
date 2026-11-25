from multiprocessing import Queue

from team_control.utils.logger import LogSaver
from team_control.vision.field import GeometryData
from team_control.vision.frame import Frame
from team_control.world.model import WorldModel


def wm_runner(wm:WorldModel,vision_q:Queue,gc_q:Queue,interval:int=5):
    logs = LogSaver() 
    while True:
        try:
            if not vision_q.empty():
                item = vision_q.get_nowait()
                if isinstance(item,Frame):
                    wm.add_new_frame(item)
                elif isinstance(item,GeometryData):
                    wm.update_geometry(item)
                else :
                    logs.warning(f"Recv unexpected item : {item}")
                    
            if not gc_q.empty():
                packet = gc_q.get_nowait()
                wm.update_gc_data(packet)
                
            
        except Exception as e:
            logs.error(f"Unexpected Exception caught, {e}")