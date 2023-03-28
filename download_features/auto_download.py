# https://nitratine.net/blog/post/python-auto-clicker/

import time
import threading
from pynput.mouse import Button, Controller
from pynput.keyboard import Listener, KeyCode, Key
from pynput.keyboard import Controller as board
import shutil

import os
down_path = "/home/marc/Downloads"
tar_path = "/home/marc/Documents/data/UCF"

down_load_button = (1281, 876)


button = Button.left
start_stop_key = KeyCode(char='s')
exit_key = KeyCode(char='e')


class ClickMouse(threading.Thread):
    def __init__(self, down_path, tar_path, button, iter=0):
        super(ClickMouse, self).__init__()
        self.down_path = down_path
        self.tar_path = tar_path
        self.button = button
        self.running = False
        self.program_running = True
        self.iter = iter

    def check_down_and_move(self):
        if len(os.listdir(self.down_path)) > 1 or len(os.listdir(self.down_path)) == 0:
            return False
        else:
            file_name = os.listdir(self.down_path)[0]
            shutil.move(self.down_path + "/" + file_name, self.tar_path + "/" + file_name)
            return True



    def start_clicking(self):
        self.running = True

    def stop_clicking(self):
        self.running = False

    def exit(self):
        self.stop_clicking()
        self.program_running = False

    def run(self):
        while self.program_running:
            while self.running:
                mouse.click(self.button)
                self.iter += 1
                print("iteration:", self.iter)
                mouse.move(-150, 0)

                time.sleep(1)
                mouse.click(self.button)
                time.sleep(1)

                kb.press(Key.right)
                kb.release(Key.right)

                time.sleep(1)
                mouse.move(150, 0)
                time.sleep(1)

                while self.running:
                    time.sleep(0.1)
                    if self.check_down_and_move():
                        time.sleep(2)
                        break

            time.sleep(0.1)

# move
# positions

kb = board()
mouse = Controller()
click_thread = ClickMouse(down_path, tar_path, button, iter=815)  # Normal 526
click_thread.start()

def on_press(key):
    if key == start_stop_key:
        if click_thread.running:
            click_thread.stop_clicking()
        else:
            click_thread.start_clicking()
    elif key == exit_key:
        click_thread.exit()
        listener.stop()


with Listener(on_press=on_press) as listener:
    listener.join()
