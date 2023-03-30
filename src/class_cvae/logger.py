import os
import time

class Logger:
    def __init__(self, output_dir, exp_name, print_to_console=False):
        self.log_path = os.path.join(output_dir, exp_name)
        self.print_to_console = print_to_console
        if not os.path.isdir(self.log_path):
            os.makedirs(self.log_path, exist_ok=True)

    def get_path(self):
        return self.log_path

    def get_timestamp(self):
        return time.localtime()

    def log(self, x):
        msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S', self.get_timestamp())}]: {x} \n"
        if self.print_to_console:
            print(msg)
        else:
            with open(f"{self.log_path}/out.log", "a") as f:
                f.write(msg)