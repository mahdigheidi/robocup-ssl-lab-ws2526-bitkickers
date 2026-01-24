# pylint: disable=no-member
import time
import pygame

from team_control.network.robot_command import RobotCommand
from team_control.robot.movement import RobotMovement
from team_control.utils.logger import LogSaver
from team_control.world.model import WorldModel
from team_control.network.sender import GrSimSender


class RCProcess():
    def __init__(self, wm: WorldModel, robot_id: int, robot_ip: str = "127.0.0.1"):
        self.robot_id = robot_id
        self.wm: WorldModel = wm
        self.last_version = 0
        self.sender = GrSimSender(ip=robot_ip, port=20011)
        self.logs = LogSaver(process_name="rc_Process", id=robot_id)

    def run(self):
        if self.robot_id is None:
            self.logs.error("ROBOT ID IS NONE EXITING . . .")
        last_update = time.time()
        vx, vy, w, k, d = 0, 0, 0, 0, 0
        while True:
            current_version: int = self.wm.get_version()
            # print(current_version)
            if current_version > self.last_version:
                self.logs.debug(
                    f"{time.time() - last_update}, {self.wm.get_latest_frame().frame_number}")
                last_update = time.time()
                self.last_version = current_version
                try:
                    robot_pos = self.wm.get_our_robots(
                        us=False, robot_id=self.robot_id).position
                    ball = self.wm.get_latest_frame().ball.position
                except Exception as e:
                    robot_pos = 0, 0, 0
                    ball = 0, 0
                    print(e)

                # vx, vy, w = calculate the vx, vy, w

            cmd = RobotCommand(0, 3, -2, 0, 0, 0)
            self.logs.info(cmd)
            self.sender.send_command(cmd)

            # command = self.run_remote_control()
            # if command.vx == 0 and command.vy == 0 and command.w ==0:
            #     command = None
            # if command is not None:
            #     self.sender.send(command)

            # if robot_pos is not None:
            # pos_relative_to_robot = world2robot(robot_position=robot_pos, target_position=ball)

            # vx,vy,w= RobotMovement.velocity_to_target(robot_pos=robot_pos,target=ball)
            # w = RobotMovement.turn_to_target(pos_relative_to_robot,speed=2)
            # print(robot_pos)

    def run_remote_control(self):
        pygame.init()
        screen = pygame.display.set_mode((400, 300))

        speed = 1
        while True:
            vx, vy, w, k, d = 0, 0, 0, 0, 0
            pygame.event.pump()  # Process internal events

            keys = pygame.key.get_pressed()  # Get key states
            if keys[pygame.K_ESCAPE]:  # Close the program
                break

            if keys[pygame.K_w] or keys[pygame.K_UP]:
                vy += speed
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                vy -= speed

            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                vx -= speed
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                vx += speed

            if keys[pygame.K_q] or keys[pygame.K_PAGEUP]:
                w += +speed
            if keys[pygame.K_e] or keys[pygame.K_PAGEDOWN]:
                w += -speed

            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                vx = vx*10
                vy = vy*10
                w = w*5

            if keys[pygame.K_SPACE]:
                k = 1
            # if keys[pygame.K_r]: #chip kick
            #     k = 2
            # if keys[pygame.K_SPACE]:
            #     d = 1

            if vx != 0 or vy != 0 or w != 0 or k != 0 or d != 0:
                cmd = RobotCommand(robot_id=self.robot_id,
                                   vx=vx, vy=vy, w=w, kick=k, dribble=d)
                self.logs.info(f"sent {cmd=}")
                self.sender.send_command(cmd)


if __name__ == "__main__":
    wm = WorldModel()
    remote_contorller = RCProcess(wm=wm, robot_id=0)
    remote_contorller.run_remote_control()
