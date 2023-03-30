import datetime
import json
import os
import shutil

import torch

DEBUG = "debug"

class Logger:
    def __init__(self, log_output="console", save_path="", exp_name=""):
        self.save_path = save_path
        self.log_output = log_output
        self.exp_name = exp_name
        if self.exp_name == "":
            self.exp_name = self.get_dt().replace(" ", "_")
            self.exp_name = self.exp_name.replace(":", "_")
            self.exp_name = self.exp_name.replace(",", "")
            self.exp_name = self.exp_name.replace("|", "")

        path = os.path.join(save_path, self.exp_name)

        # Debug is meant for repeated tries with no intention of saving the results for later
        if self.exp_name == DEBUG and os.path.exists(path):
            shutil.rmtree(path)

        if not os.path.exists(path):
            os.makedirs(path)
        else:
            i = 1
            while os.path.exists(f"{path}_{i}"):
                i += 1
            self.exp_name += f"_{i}"
            os.makedirs(os.path.join(save_path, self.exp_name))

    def get_dt(self):
        dt = datetime.datetime.now()
        dt_format = dt.strftime("%b %d, %Y | %H:%M:%S")
        return dt_format

    def log(self, txt):
        dt_format = self.get_dt()
        line = f"[{dt_format}]: {txt}"

        if self.log_output == "console":
            print(line)
        elif self.log_output == "file":
            with open(os.path.join(self.save_path, self.exp_name, "out.log"), "a") as f:
                f.write(f"{line}\n")

    def save_json(self, data, filename):
        if self.log_output == "none": return
        with open(os.path.join(self.save_path, self.exp_name, filename), 'w') as f:
            json.dump(data, f)

    def get_save_dir(self):
        return os.path.join(self.save_path, self.exp_name)

    def save_model(self, model, name="model.pt"):
        torch.save(model.state_dict(), os.path.join(self.save_path, self.exp_name, name))
