from torch.utils.data import Dataset
import trajnetplusplustools

class TrajNetData(Dataset):
    def __init__(self, train_scenes):
        self.scenes=train_scenes

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, item):
        breakpoint()
        (filename, scene_id, paths) = self.scenes[item]
        scene = trajnetplusplustools.Reader.paths_to_xy(paths) # num frames, # peds, (x,y)
        return scene