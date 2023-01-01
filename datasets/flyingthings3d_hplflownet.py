import os
import os.path as osp
import glob
import numpy as np
from .generic import SceneFlowDataset


class FT3D(SceneFlowDataset):
    def __init__(self, root_dir, nb_points, mode):
        """
        Construct the FlyingThing3D datatset as in:
        Gu, X., Wang, Y., Wu, C., Lee, Y.J., Wang, P., HPLFlowNet: Hierarchical
        Permutohedral Lattice FlowNet for scene ﬂow estimation on large-scale
        point clouds. IEEE Conf. Computer Vision and Pattern Recognition
        (CVPR). pp. 3254–3263 (2019)

        Parameters
        ----------
        root_dir : str
            Path to root directory containing the datasets.
        nb_points : int
            Maximum number of points in point clouds.
        mode : str
            'train': training dataset.

            'val': validation dataset.

            'test': test dataset

        """

        super(FT3D, self).__init__(nb_points)

        self.mode = mode
        self.root_dir = root_dir
        self.filenames = self.get_file_list()

    def __len__(self):

        return len(self.filenames)

    def get_file_list(self):
        """
        Find and filter out paths to all examples in the dataset.

        """

        # Get list of filenames / directories
        if self.mode == "train_all":
            pattern = "train/0*"
        elif self.mode == "train" or self.mode == "val":
            pattern = "train/0*"
        elif self.mode == "test":
            # pattern = "val/0*"
            pattern = "test/0*"
        else:
            raise ValueError("Mode " + str(self.mode) + " unknown.")
        filenames = glob.glob(os.path.join(self.root_dir, pattern))

        
        if self.mode == "train_all":
            assert len(filenames) == 19640, "Problem with size of training set"
            filenames = np.sort(filenames)
        # Train / val / test split
        elif self.mode == "train" or self.mode == "val":
            assert len(filenames) == 19640, "Problem with size of training set"
            ind_val = set(np.linspace(0, 19639, 2000).astype("int"))
            ind_all = set(np.arange(19640).astype("int"))
            ind_train = ind_all - ind_val
            assert (
                    len(ind_train.intersection(ind_val)) == 0
            ), "Train / Val not split properly"
            filenames = np.sort(filenames)
            if self.mode == "train":
                filenames = filenames[list(ind_train)]
            elif self.mode == "val":
                filenames = filenames[list(ind_val)]
        else:
            assert len(filenames) == 3824, "Problem with size of test set"
       
        return list(filenames)
        
    def load_sequence(self, idx):
        """
        Load a sequence of point clouds.

        Parameters
        ----------
        idx : int
            Index of the sequence to load.

        Returns
        -------
        sequence : list(np.array, np.array)
            List [pc1, pc2] of point clouds between which to estimate scene
            flow. pc1 has size n x 3 and pc2 has size m x 3.

        ground_truth : list(np.array, np.array)
            List [mask, flow]. mask has size n x 1 and pc1 has size n x 3.
            flow is the ground truth scene flow between pc1 and pc2. mask is
            binary with zeros indicating where the flow is not valid/occluded.

        """

        # Load data
        sequence = []  # [Point cloud 1, Point cloud 2]

        data = np.load(self.filenames[idx])
        for fname in ["pc1", "pc2"]:
            pc = data[fname]
            #　['pc1', 'pc2', 'flow', 'inst_pc1', 'inst_pc2']
            # pc[..., 0] *= -1
            # pc[..., -1] *= -1
            sequence.append(pc)


        ground_truth = [
            np.ones_like(sequence[0][:, 0:1]),
            data["flow"],
        ]  # [Occlusion mask, flow]

        return sequence, ground_truth





    def __init__(self, root_dir,nb_points,mode):
        super(FlyingThings3DSubset_Occlusion, self).__init__(nb_points)

        self.mode = mode
        self.root_dir = root_dir
        self.filenames = self.get_file_list()

        #self.root = root_dir

        # self.train = train
        self.num_points = nb_points
        self.size = len(self.filenames)

        
        # self.cache = {}
        # self.cache_size = 30000

        

    def __len__(self):
        return len(self.filenames)


    def get_file_list(self):
        if "train" in self.mode:
            self.datapath = glob.glob(os.path.join(self.root_dir, "TRAIN*.npz"))
        else:
            self.datapath = glob.glob(os.path.join(self.root_dir, "TEST*.npz"))

        ###### deal with one bad datapoint with nan value
        filenames = [
            d for d in self.datapath if "TRAIN_C_0140_left_0006-0" not in d
        ]
        ######

        out_files = []

        for pc_filepath in filenames:
            with open(pc_filepath, "rb") as fp:
                data = np.load(fp)
                flow = data["flow"]
                mask1 = data["valid_mask1"].reshape(flow.shape[0],1)
                mask_01 = mask1.astype(int)
                mask_sum = np.sum(mask_01)
                # if mask_sum > (0.9*self.nb_points):
                # if (mask_sum > 1) and mask_sum < (0.9*self.nb_points):
                if mask_sum > 1:
                    out_files.append(pc_filepath)
        
        #return list(filenames)
        return out_files


    def load_sequence(self, index):
        index_origin = index
        # if index in self.cache:
        #     pos1, pos2, color1, color2, flow, mask1 = self.cache[index]
        # else:
        #     fn = self.filenames[index]
        #     with open(fn, "rb") as fp:
        #         data = np.load(fp)
        #         pos1 = data["points1"]
        #         pos2 = data["points2"]
        #         color1 = data["color1"] / 255
        #         color2 = data["color2"] / 255
        #         flow = data["flow"]
        #         mask1 = data["valid_mask1"]

        #     if len(self.cache) < self.cache_size:
        #         self.cache[index] = (pos1, pos2, color1, color2, flow, mask1)
        sequence = []
        ground_truth =[]

        fn = self.filenames[index]
        with open(fn, "rb") as fp:
            data = np.load(fp)
            pos1 = data["points1"]
            pos2 = data["points2"]
            #color1 = data["color1"] / 255
            #color2 = data["color2"] / 255
            flow = data["flow"]
            mask1 = data["valid_mask1"].reshape(flow.shape[0],1)
            # mask_01 = mask1.astype(int)
            # mask_sum = np.sum(mask_01)


            
        # while (mask_sum < 0.9*self.nb_points):
        #     index+=1
        #     if index >= self.size:
        #         break
        #     fn = self.filenames[index]
        #     with open(fn, "rb") as fp:
        #         data = np.load(fp)
        #         pos1 = data["points1"]
        #         pos2 = data["points2"]
        #         #color1 = data["color1"] / 255
        #         #color2 = data["color2"] / 255
        #         flow = data["flow"]
        #         mask1 = data["valid_mask1"].reshape(flow.shape[0],1)
        #         mask_01 = mask1.astype(int)
        #         mask_sum = np.sum(mask_01)

        # if index >= self.size:
        #     index= index_origin
        #     while (mask_sum < 0.9*self.nb_points):
        #         index-=1
        #         fn = self.filenames[index]
        #         with open(fn, "rb") as fp:
        #             data = np.load(fp)
        #             pos1 = data["points1"]
        #             pos2 = data["points2"]
        #             #color1 = data["color1"] / 255
        #             #color2 = data["color2"] / 255
        #             flow = data["flow"]
        #             mask1 = data["valid_mask1"].reshape(flow.shape[0],1)
        #             mask_01 = mask1.astype(int)
        #             mask_sum = np.sum(mask_01)
        sequence.append(pos1)
        sequence.append(pos2)
        ground_truth.append(mask1)
        ground_truth.append(flow)




        # if self.train:
        #     n1 = pos1.shape[0]
        #     sample_idx1 = np.random.choice(n1, self.num_points, replace=False)
        #     n2 = pos2.shape[0]
        #     sample_idx2 = np.random.choice(n2, self.num_points, replace=False)

        #     pos1_ = np.copy(pos1[sample_idx1, :])
        #     pos2_ = np.copy(pos2[sample_idx2, :])
        #     color1_ = np.copy(color1[sample_idx1, :])
        #     color2_ = np.copy(color2[sample_idx2, :])
        #     flow_ = np.copy(flow[sample_idx1, :])
        #     mask1_ = np.copy(mask1[sample_idx1])
        # else:
        #     pos1_ = np.copy(pos1[: self.num_points, :])
        #     pos2_ = np.copy(pos2[: self.num_points, :])
        #     color1_ = np.copy(color1[: self.num_points, :])
        #     color2_ = np.copy(color2[: self.num_points, :])
        #     flow_ = np.copy(flow[: self.num_points, :])
        #     mask1_ = np.copy(mask1[: self.num_points])

        return sequence, ground_truth