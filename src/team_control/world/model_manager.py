from multiprocessing.managers import BaseManager

from team_control.world.model import WorldModel


class WorldModelManager(BaseManager):
    pass


WorldModelManager.register("WorldModel", WorldModel)
