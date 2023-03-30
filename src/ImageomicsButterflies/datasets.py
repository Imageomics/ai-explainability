from torch.utils.data import Dataset
from data_tools import handle_image_list, handle_image_folder, rgb_img_loader

class ImageList(Dataset):
    def __init__(self, image_list, transform=None):
        self.paths, self.labels, self.path_label_map = handle_image_list(image_list)
        self.transform = transform
        self.loader = rgb_img_loader

    def get_label(self, path):
        if path not in self.path_label_map:
            return None
        
        return self.path_label_map[path]
    
    def get_num_classes(self):
        return max(self.labels) + 1

    def load_img(self, path):
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __getitem__(self, index):
        path = self.paths[index]
        lbl = self.labels[index]
        img = self.load_img(path)

        return img, lbl, path

    def __len__(self):
        return len(self.paths)


class ImageFolder(Dataset):
    def __init__(self, img_dir, transform=None):
        self.paths, self.labels, self.path_label_map, self.class_names = handle_image_folder(img_dir)
        self.transform = transform
        self.loader = rgb_img_loader

    def get_label(self, path):
        if path not in self.path_label_map:
            return None
        
        return self.path_label_map[path]
    
    def get_num_classes(self):
        return max(self.labels) + 1
    
    def get_class_names(self):
        return self.class_names

    def load_img(self, path):
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __getitem__(self, index):
        path = self.paths[index]
        lbl = self.labels[index]
        img = self.load_img(path)

        return img, lbl, path

    def __len__(self):
        return len(self.paths)