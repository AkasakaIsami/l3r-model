import os.path
from typing import Optional, Callable, Union, List, Tuple

import torch
from torch_geometric.data import InMemoryDataset


class SingleProjectDataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None, project=None, dataset_type="train", methods=None):
        self.project = project
        self.methods = methods

        super(SingleProjectDataset, self).__init__(root, transform, pre_transform)

        if dataset_type == "train":
            print(f"{dataset_type} using {self.processed_paths[0]} as dataset")
            self.data, self.slices = torch.load(self.processed_paths[0])

        elif dataset_type == "validate":
            print(f"{dataset_type} using {self.processed_paths[1]} as dataset")
            self.data, self.slices = torch.load(self.processed_paths[1])

        elif dataset_type == "test":
            print(f"{dataset_type} using {self.processed_paths[2]} as dataset")
            self.data, self.slices = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        paths = []
        for item in self.methods.values:
            clz = item[0]
            method = item[1]
            path = os.path.join(self.project, clz, method)
            paths.append(path)

        return paths

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        processed_train_path = os.path.join(self.project, "train")
        processed_dev_path = os.path.join(self.project, "dev")
        processed_test_path = os.path.join(self.project, "test")
        return [processed_train_path, processed_dev_path, processed_test_path]

    def download(self):
        pass

    def process(self):
        for path in self.raw_paths:
            files = os.listdir(path)

        train_datalist = []

        validate_datalist = []

        test_datalist = []

        # data = Data(
        #     x=None,
        #     edge_index=None,
        #     y=None,
        #     statements=[[1, 0], [0, 1], [1, 0]],
        #     edges=[[[1, 2, 3], [2, 3, 4]], ]
        # )

        print("collating train data")
        data, slices = self.collate(train_datalist)
        torch.save((data, slices), self.processed_paths[0])

        print("collating validate data")
        data, slices = self.collate(validate_datalist)
        torch.save((data, slices), self.processed_paths[1])

        print("collating test data")
        data, slices = self.collate(test_datalist)
        torch.save((data, slices), self.processed_paths[2])


class AllProjectsDataset(InMemoryDataset):

    def __init__(self, root: Optional[str] = None, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None, log: bool = True):
        super().__init__(root, transform, pre_transform, pre_filter, log)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return []

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return []

    def download(self):
        pass

    def process(self):
        pass
