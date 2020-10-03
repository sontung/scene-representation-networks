import os
import torch
import numpy as np
from glob import glob
import data_util
import util
import torchvision
import pickle
import random
import json
from PIL import Image
from os.path import isfile, join
from os import listdir
from torch.utils.data import DataLoader


def pick(list, item_idcs):
    if not list:
        return list
    return [list[i] for i in item_idcs]


class SceneInstanceDataset():
    """This creates a dataset class for a single object instance (such as a single car)."""

    def __init__(self,
                 instance_idx,
                 instance_dir,
                 specific_observation_idcs=None,  # For few-shot case: Can pick specific observations only
                 img_sidelength=None,
                 num_images=-1):
        self.instance_idx = instance_idx
        self.img_sidelength = img_sidelength
        self.instance_dir = instance_dir

        color_dir = os.path.join(instance_dir, "rgb")
        pose_dir = os.path.join(instance_dir, "pose")
        param_dir = os.path.join(instance_dir, "params")

        if not os.path.isdir(color_dir):
            print("Error! root dir %s is wrong" % instance_dir)
            return

        self.has_params = os.path.isdir(param_dir)
        self.color_paths = sorted(data_util.glob_imgs(color_dir))
        self.pose_paths = sorted(glob(os.path.join(pose_dir, "*.txt")))

        if self.has_params:
            self.param_paths = sorted(glob(os.path.join(param_dir, "*.txt")))
        else:
            self.param_paths = []

        if specific_observation_idcs is not None:
            self.color_paths = pick(self.color_paths, specific_observation_idcs)
            self.pose_paths = pick(self.pose_paths, specific_observation_idcs)
            self.param_paths = pick(self.param_paths, specific_observation_idcs)
        elif num_images != -1:
            idcs = np.linspace(0, stop=len(self.color_paths), num=num_images, endpoint=False, dtype=int)
            self.color_paths = pick(self.color_paths, idcs)
            self.pose_paths = pick(self.pose_paths, idcs)
            self.param_paths = pick(self.param_paths, idcs)

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        self.img_sidelength = new_img_sidelength

    def __len__(self):
        return len(self.pose_paths)

    def __getitem__(self, idx):
        intrinsics, _, _, _ = util.parse_intrinsics(os.path.join(self.instance_dir, "intrinsics.txt"),
                                                    trgt_sidelength=self.img_sidelength)
        intrinsics = torch.Tensor(intrinsics).float()
        rgb = data_util.load_rgb(self.color_paths[idx], sidelength=self.img_sidelength)
        print("1",rgb.shape)

        rgb = rgb.reshape(3, -1).transpose(1, 0)
        print(rgb.shape)

        pose = data_util.load_pose(self.pose_paths[idx])

        if self.has_params:
            params = data_util.load_params(self.param_paths[idx])
        else:
            params = np.array([0])

        uv = np.mgrid[0:self.img_sidelength, 0:self.img_sidelength].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).long()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "instance_idx": torch.Tensor([self.instance_idx]).squeeze(),
            "rgb": torch.from_numpy(rgb).float(),
            "pose": torch.from_numpy(pose).float(),
            "uv": uv,
            "param": torch.from_numpy(params).float(),
            "intrinsics": intrinsics
        }
        return sample


class SceneClassDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 root_dir,
                 img_sidelength=None,
                 max_num_instances=-1,
                 max_observations_per_instance=-1,
                 specific_observation_idcs=None,  # For few-shot case: Can pick specific observations only
                 samples_per_instance=2):

        self.samples_per_instance = samples_per_instance
        self.instance_dirs = sorted(glob(os.path.join(root_dir, "*/")))

        assert (len(self.instance_dirs) != 0), "No objects in the data directory"

        if max_num_instances != -1:
            self.instance_dirs = self.instance_dirs[:max_num_instances]

        self.all_instances = [SceneInstanceDataset(instance_idx=idx,
                                                   instance_dir=dir,
                                                   specific_observation_idcs=specific_observation_idcs,
                                                   img_sidelength=img_sidelength,
                                                   num_images=max_observations_per_instance)
                              for idx, dir in enumerate(self.instance_dirs)]

        self.num_per_instance_observations = [len(obj) for obj in self.all_instances]
        self.num_instances = len(self.all_instances)

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        for instance in self.all_instances:
            instance.set_img_sidelength(new_img_sidelength)

    def __len__(self):
        return np.sum(self.num_per_instance_observations)

    def get_instance_idx(self, idx):
        """Maps an index into all tuples of all objects to the idx of the tuple relative to the other tuples of that
        object
        """
        obj_idx = 0
        while idx >= 0:
            idx -= self.num_per_instance_observations[obj_idx]
            obj_idx += 1
        return obj_idx - 1, int(idx + self.num_per_instance_observations[obj_idx - 1])

    def collate_fn(self, batch_list):
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            # make them all into a new dict
            ret = {}
            for k in entry[0][0].keys():
                ret[k] = []
            # flatten the list of list
            for b in entry:
                for k in entry[0][0].keys():
                    ret[k].extend( [bi[k] for bi in b])
            for k in ret.keys():
                if type(ret[k][0]) == torch.Tensor:
                    ret[k] = torch.stack(ret[k])
            all_parsed.append(ret)

        return tuple(all_parsed)

    def __getitem__(self, idx):
        """Each __getitem__ call yields a list of self.samples_per_instance observations of a single scene (each a dict),
        as well as a list of ground-truths for each observation (also a dict)."""
        obj_idx, rel_idx = self.get_instance_idx(idx)

        observations = []
        observations.append(self.all_instances[obj_idx][rel_idx])

        for i in range(self.samples_per_instance - 1):
            observations.append(self.all_instances[obj_idx][np.random.randint(len(self.all_instances[obj_idx]))])

        ground_truth = [{'rgb':ray_bundle['rgb']} for ray_bundle in observations]

        return observations, ground_truth


class PBWDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir="/home/sontung/thesis/photorealistic-blocksworld/blocks-4-3",
                 train=True, train_size=0.6, nb_samples=-1, if_save_data=True):
        print("Loading from", root_dir)
        super(PBWDataset, self).__init__()
        self.root_dir = root_dir
        self.train = train
        identifier = root_dir.split("/")[-1]
        json_dir = "%s/scene" % root_dir

        self.image_dir = "%s/image" % root_dir
        self.transform = torchvision.transforms.ToTensor()
        self.transform_to_pil = torchvision.transforms.ToPILImage()

        self.load_json(identifier, json_dir, if_save_data)

        self.json2im = self.load_json2im(nb_samples=nb_samples)

        keys = list(self.json2im.keys())
        if train:
            self.data = {du3: self.json2im[du3] for du3 in keys[:int(len(keys)*train_size)]}
        else:
            self.data = {du3: self.json2im[du3] for du3 in keys[int(len(keys)*train_size):]}
        self.keys = list(self.data.keys())
        print("loaded", len(self.json2im))
        self.num_instances = len(self.data)

    def load_json(self, identifier, json_dir, if_save_data):
        self.scene_jsons = [join(json_dir, f) for f in listdir(json_dir)
                            if isfile(join(json_dir, f))]
        if isfile("data/json2sg-%s" % identifier):
            print("Loading precomputed json2sg:", "data/json2sg-%s" % identifier)
            with open("data/json2sg-%s" % identifier, 'rb') as f:
                self.json2sg = pickle.load(f)
        else:
            self.json2sg = {}
            for js in self.scene_jsons:
                self.json2sg[js] = read_scene_json(js)
            if if_save_data:
                with open("data/json2sg-%s" % identifier, 'wb') as f:
                    pickle.dump(self.json2sg, f, pickle.HIGHEST_PROTOCOL)

    def load_json2im(self, nb_samples=1000):
        if nb_samples < 0:
            nb_samples = len(self.scene_jsons)
        name = "%s-%d" % (self.root_dir.split("/")[-1], nb_samples)
        if isfile("data/%s" % name):
            print("Loading precomputed json2im:", "data/%s" % name)
            with open("data/%s" % name, 'rb') as f:
                return pickle.load(f)
        else:
            res_dict = {}
            if nb_samples > 0:
                random.shuffle(self.scene_jsons)
            for item in range(len(self.scene_jsons))[:nb_samples]:
                information = self.json2sg[self.scene_jsons[item]]  # {imID, objects: [color, pix coord, location, box, ext]}

                img_pil = Image.open("%s/%s" % (self.image_dir, information["image_id"])).convert('RGB')
                img = self.transform(img_pil).unsqueeze(0)

                def_ext_mat = information["objects"][0][-1]
                all_info = []
                for obj in information["objects"]:
                    all_info.append(obj[:-1])
                    assert torch.sum(obj[-1]-def_ext_mat).item() < 0.000001

                res_dict[self.scene_jsons[item]] = (img, def_ext_mat, all_info)
            with open("data/%s" % name, 'wb') as f:
                pickle.dump(res_dict, f, pickle.HIGHEST_PROTOCOL)
            return res_dict

    def __len__(self):
        return len(self.keys)

    def collate_fn(self, batch):
        all_imgs, all_ext_mat, all_info = [], [], []
        all_img_mat = []
        for i, (img, def_ext_mat, info) in enumerate(batch):
            all_img_mat.append(img)
            img = img.reshape(3, -1).transpose(1, 0).unsqueeze(0)
            all_imgs.append(img)
            all_ext_mat.append(def_ext_mat.unsqueeze(0))
            all_info.append(info)

        all_img_mat = torch.cat(all_img_mat)
        all_imgs = torch.cat(all_imgs)
        all_ext_mat = torch.cat(all_ext_mat)

        return all_imgs, all_ext_mat, all_info, all_img_mat

    def __getitem__(self, item):
        return self.data[self.keys[item]]

def read_scene_json(json_file_dir):

    id2color = {
        "gray": [87, 87, 87],
        "red": [173, 35, 35],
        "blue": [42, 75, 215],
        "green": [29, 105, 20],
        "brown": [129, 74, 25],
        "purple": [129, 38, 192],
        "cyan": [41, 208, 208],
        "yellow": [255, 238, 51],
        "c1": [42, 87, 9],
        "c2": [255, 102, 255],
        "orange": [255, 140, 0]
    }
    color2id = {tuple(v): u for u, v in id2color.items()}
    with open(json_file_dir, 'r') as json_file:
        du = json.load(json_file)

    returned_json = {"image_id": du["image_filename"], "objects": []}
    def_frame = torch.tensor(
        [
            [-0.0914, -0.0914,  0.2000],
            [-0.0914,  0.0914,  0.2000],
            [ 0.0914,  0.0914,  0.2000]
        ]
    )

    for obj in du["objects"]:
        infor = []
        color = tuple([int(du33*255) for du33 in obj["color"]][:-1])
        object_id = color2id[color]
        infor.append(object_id)
        infor.append(obj["pixel_coords"])
        infor.append(obj["location"])
        infor.append(obj["bbox"])
        infor.append(torch.tensor(obj["recon_data"]["matrix_world"]))
        assert torch.sum(torch.tensor(obj["recon_data"]["frame"])-def_frame).item()<=0.001
        returned_json["objects"].append(infor)

    return returned_json # id location box ext

def recon_sg(obj_names, locations, if_return_assigns=False):
    """
    reconstruct a sg from object names and coordinates
    """
    location_dict = {}
    objects = []

    if type(locations) == torch.Tensor:
        locations = locations.cpu().numpy()
    elif isinstance(locations, list):
        locations = np.array(locations)

    locations = locations.reshape(-1, 2)
    k_means_assign = kmeans(locations[:, 0])

    for idx, object_id in enumerate(obj_names):
        a_key = k_means_assign[idx]
        if a_key not in location_dict:
            location_dict[a_key] = [(object_id, locations[idx][1])]
        else:
            location_dict[a_key].append((object_id, locations[idx][1]))
        objects.append(object_id)
    relationships = [
        ["brown", "left", "purple"],
        ["purple", "left", "cyan"],
    ]
    for du3 in location_dict:
        location = sorted(location_dict[du3], key=lambda x: x[1])
        while len(location) > 1:
            o1 = location.pop()[0]
            o2 = location[-1][0]
            relationships.append([o1, "up", o2])
    if if_return_assigns:
        return relationships, k_means_assign
    return relationships

def recon_sg2(json_file_dir, if_add_bases=True):
    """
    reconstruct a sg from a scene json file
    """
    id2color = {
        "gray": [87, 87, 87],
        "red": [173, 35, 35],
        "blue": [42, 75, 215],
        "green": [29, 105, 20],
        "brown": [129, 74, 25],
        "purple": [129, 38, 192],
        "cyan": [41, 208, 208],
        "yellow": [255, 238, 51],
        "c1": [42, 87, 9],
        "c2": [255, 102, 255],
        "orange": [255, 140, 0]
    }

    color2id = {tuple(v): u for u, v in id2color.items()}
    with open(json_file_dir, 'r') as json_file:
        du = json.load(json_file)
    location_dict = {}
    objects = []
    bboxes = []
    for obj in du["objects"]:
        color = tuple([int(du33*255) for du33 in obj["color"]][:-1])
        object_id = color2id[color]
        a_key = "%.3f" % obj["location"][0]
        if a_key not in location_dict:
            location_dict[a_key] = [(object_id, obj["location"][2])]
        else:
            location_dict[a_key].append((object_id, obj["location"][2]))
        objects.append(object_id)
        bboxes.append([
            obj["bbox"][0]/128.0,
            obj["bbox"][1]/128.0,
            obj["bbox"][2]/128.0,
            obj["bbox"][3]/128.0,
            ])
    obj2id = {objects[du4]: objects[du4] for du4 in range(len(objects))}
    if if_add_bases:
        relationships = [
            [obj2id["brown"], "left", obj2id["purple"]],
            [obj2id["purple"], "left", obj2id["cyan"]],
        ]
    else:
        relationships = []
    for du3 in location_dict:
        location = sorted(location_dict[du3], key=lambda x: x[1])
        while len(location) > 1:
            o1 = location.pop()[0]
            o2 = location[-1][0]
            relationships.append([obj2id[o1], "up", obj2id[o2]])
            assert o1 not in ["cyan", "purple", "brown"]

    return relationships

if __name__ == '__main__':
    d = PBWDataset()
    train_dataloader = DataLoader(d,
                                  batch_size=16,
                                  shuffle=True,
                                  collate_fn=d.collate_fn)
    for b in train_dataloader:
        print(b[0].size())
        print(b[1].size())
        print(len(b[2]))
        break