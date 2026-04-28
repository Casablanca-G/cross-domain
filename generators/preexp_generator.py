# generators/preexp_generator.py
import numpy as np
from tqdm import tqdm
from generators.generator import CDSRRegSeq2SeqGeneratorUser
from zujian.utils import unzip_data, concat_data
from generators.data import CDSRRegSeq2SeqDatasetUser, CDSREvalSeq2SeqDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


class PreExpGenerator(CDSRRegSeq2SeqGeneratorUser):
    """
    预实验数据生成器：仅保留在两个领域均有交互的重叠用户
    """

    def __init__(self, args, logger, device):
        super().__init__(args, logger, device)
        self._filter_overlapping_users()
        self.logger.info(f"[PreExp] After filtering, {len(self.train)} overlapping users remain.")

    def _filter_overlapping_users(self):
        """只保留在域 A 和域 B 均有交互的用户"""
        filtered_train = {}
        filtered_domain_train = {}
        for user in self.train.keys():
            domains = set(self.domain_train[user])
            if 0 in domains and 1 in domains:  # 两个域都有交互
                filtered_train[user] = self.train[user]
                filtered_domain_train[user] = self.domain_train[user]
        self.train = filtered_train
        self.domain_train = filtered_domain_train
        # 同步过滤 valid / test（只保留重叠用户）
        filtered_valid = {u: v for u, v in self.valid.items() if u in self.train}
        filtered_domain_valid = {u: v for u, v in self.domain_valid.items() if u in self.train}
        filtered_test = {u: v for u, v in self.test.items() if u in self.train}
        filtered_domain_test = {u: v for u, v in self.domain_test.items() if u in self.train}
        self.valid = filtered_valid
        self.domain_valid = filtered_domain_valid
        self.test = filtered_test
        self.domain_test = filtered_domain_test