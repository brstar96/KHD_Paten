import os

# from nsml import DATASET_PATH # NSML에서 테스트할때 주석풀고 클라우드 경로 받아오기.
local_parent_dir = os.path.abspath(os.getcwd() + "/../../")

class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'LocalDataset':
            return os.path.join(local_parent_dir, 'localDatasetPath')
        # elif dataset == 'KHD_NSML':
        #     return DATASET_PATH
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
