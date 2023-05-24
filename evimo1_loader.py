import torch
import cv2
import os
import torch.nn.functional as F
import numpy as np
from torchvision.utils import make_grid
import h5py

class EVIMO_Sequence(torch.utils.data.Dataset):
    def __init__(self, seq_path: str,
                        dt=0.020,
                        args=None
                        ):

        self.args = args
        self.seq_path = seq_path
        self.dt = dt    # interval of events slice[s]

        seq_path = seq_path.replace(".npz", ".h5")
        seq = h5py.File(seq_path, 'r')
        self.K = seq['K']

        self.x = seq['x_old']
        self.y = seq['y_old']
        self.t = seq['t_old']
        self.p = seq['p_old']

        self.res = [np.array(seq['h']).item(), np.array(seq['w']).item()]

    def denoise_filter(self, pose, thres=1.):

        '''
        pose: [N, 3]
        '''
        new_pose = pose.copy()
        for i in range(pose.shape[0]):

            xyz = pose[i, :]
            remove = np.any(np.abs(xyz) > thres)
            if (not remove):
                continue

            left = i
            select = np.all(np.abs(pose[left, :]) < thres)
            while (not select):
                left -= 1
                select = np.all(np.abs(pose[left, :]) < thres)
            left_value = pose[left]

            right = i
            select = np.all(np.abs(pose[right, :]) < thres)
            while (not select):
                right += 1
                select = np.all(np.abs(pose[right, :]) < thres)
            right_value = pose[right]

            left_weight = 1 - (i - left) / (right - left)
            new_pose[i] = left_value * left_weight + (1-left_weight) * right_value

        return new_pose

    @staticmethod
    def get_all_poses(meta, pose_key="full_trajectory"):
        vicon_pose_samples = len(meta[pose_key])
        poses = {}
        key_i = {}
        for key in meta[pose_key][0].keys():
            if key == 'id' or key == 'ts' or key == 'gt_frame' or key == 'classical_frame':
                continue
            poses[key] = np.zeros((vicon_pose_samples, 1+3+4))
            key_i[key] = 0

        # Convert camera poses to array
        for all_pose in meta[pose_key]:
            for key in poses.keys():
                if key == 'id' or key == 'ts' or key == 'gt_frame' or key == 'classical_frame':
                    continue

                if key in all_pose:
                    i = key_i[key]
                    poses[key][i, 0] = all_pose['ts']
                    poses[key][i, 1] = all_pose[key]['pos']['t']['x']
                    poses[key][i, 2] = all_pose[key]['pos']['t']['y']
                    poses[key][i, 3] = all_pose[key]['pos']['t']['z']
                    poses[key][i, 4] = all_pose[key]['pos']['q']['x']
                    poses[key][i, 5] = all_pose[key]['pos']['q']['y']
                    poses[key][i, 6] = all_pose[key]['pos']['q']['z']
                    poses[key][i, 7] = all_pose[key]['pos']['q']['w']
                    key_i[key] += 1

        for key in poses.keys():
            poses[key] = poses[key][:key_i[key], :]

        return poses
        
    def get_events(self, idx0, idx1):
        """
        Get all the events in between two indices.
        :param idx0: start index
        :param idx1: end index
        :return xs: [N] numpy array with event x location
        :return ys: [N] numpy array with event y location
        :return ts: [N] numpy array with event timestamp
        :return ps: [N] numpy array with event polarity ([0, 1])
        """
        xs = np.array(self.x[idx0:idx1])
        ys = np.array(self.y[idx0:idx1])
        # ts = (np.array(self.t[idx0:idx1]) * 1e6).astype(int)    # [s] to [us]
        ts = np.array(self.t[idx0:idx1])    #[s] 
        ps = np.array(self.p[idx0:idx1])
        # ts -= ts[0]  # chunked events starting at t0 = 0
        assert np.all(xs >= 0)
        assert np.all(ys >= 0)
        assert np.all(ps >= 0) 
        return xs, ys, ts, ps

    def find_ts_index(self, timestamp):
        t = self.t
        t = ((t - t[0]) * 1e6).astype(int)
        t_idx = np.searchsorted(t, timestamp)
        return t_idx

    def parse_events(self, xs, ys, ts, ps):
        xs = torch.from_numpy(xs.astype(np.float32))
        ys = torch.from_numpy(ys.astype(np.float32))
        ts = torch.from_numpy(ts.astype(np.float32))
        # ps = (torch.from_numpy(ps.astype(np.float32)) * 2 - 1)  # convert (0, 1) to (-1, 1)
        ps = torch.from_numpy(ps.astype(np.float32)) 
        return xs, ys, ts, ps
    
    def __len__(self):
        self.length  = (self.t - self.t[0]) // self.dt
        # self.length = len(self.frame2event) - 3
        return self.length

    def __getitem__(self, index):

        output = dict()

        idx0 = self.find_ts_index(index * self.dt)
        idx1 = self.find_ts_index((index + 1) * self.dt)
        xs, ys, ts, ps = self.get_events(idx0, idx1)
        xs, ys, ts, ps = self.parse_events(xs, ys, ts, ps)

        output['events_list'] = torch.stack([xs, ys, ts, ps], dim=-1)
        output['seq_name'] = self.seq_path

        return output
