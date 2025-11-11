import torch
import scanpy as sc
from scipy import sparse
from typing import Tuple
from torch import Tensor
from sklearn.neighbors import kneighbors_graph
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import TfidfTransformer


class TensorDataSetWithIndex(TensorDataset):
    tensors: Tuple[Tensor, ...]

    def __init__(self, *tensors: Tensor):
        super(TensorDataSetWithIndex, self).__init__(*tensors)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors), index


def prepare_dataloader(args):
    # Load and Preprocess Source (RNA) Data
    source_adata = sc.read_h5ad(args.data_path + args.source_data)
    if isinstance(source_adata.X, sparse.csr_matrix):
        source_adata.X = source_adata.X.toarray()
    if args.source_preprocess == "Standard":
        sc.pp.normalize_total(source_adata, target_sum=1e4)
        sc.pp.log1p(source_adata)
    elif args.source_preprocess == "TFIDF":
        tfidf = TfidfTransformer()
        source_adata.X = tfidf.fit_transform(source_adata.X).toarray()
    else:
        raise NotImplementedError
    sc.pp.scale(source_adata)
    source_adata.obs["Domain"] = args.source_data[:-5]
    source_label = source_adata.obs["CellType"]
    source_label_int = source_label.rank(method="dense", ascending=True).astype(int) - 1
    source_label = source_label.values
    source_label_int = source_label_int.values
    label_map = dict()
    for k in range(source_label_int.max() + 1):
        label_map[k] = source_label[source_label_int == k][0]

    # Load and Preprocess Target (ATAC) Data
    target_adata = sc.read_h5ad(args.data_path + args.target_data)
    if isinstance(target_adata.X, sparse.csr_matrix):
        target_adata.X = target_adata.X.toarray()
    if args.target_preprocess == "Standard":
        sc.pp.normalize_total(target_adata, target_sum=1e4)
        sc.pp.log1p(target_adata)
    elif args.target_preprocess == "TFIDF":
        tfidf = TfidfTransformer()
        target_adata.X = tfidf.fit_transform(target_adata.X).toarray()
    else:
        raise NotImplementedError
    sc.pp.scale(target_adata)
    target_adata.obs["Domain"] = args.target_data[:-5]

    # Prepare PyTorch Data
    source_data = torch.from_numpy(source_adata.X).float()
    source_label_int = torch.from_numpy(source_label_int).long()
    target_data = torch.from_numpy(target_adata.X).float()
    target_index = torch.arange(target_data.shape[0]).long()

    # Prepare PyTorch Dataset and DataLoader
    source_dataset = TensorDataset(source_data, source_label_int)
    source_dataloader_train = DataLoader(
        dataset=source_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    source_dataloader_eval = DataLoader(
        dataset=source_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    target_dataset = TensorDataSetWithIndex(target_data, target_index)
    target_dataloader_train = DataLoader(
        dataset=target_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    target_dataloader_eval = DataLoader(
        dataset=target_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    gene_num = source_data.shape[1]
    type_num = torch.unique(source_label_int).shape[0]

    print("Data Loaded with the Following Configurations:")
    print(
        "Source data:",
        args.source_data[:-5],
        "\tPreprocess:",
        args.source_preprocess,
        "\tShape",
        list(source_data.shape),
    )
    print(
        "Target data:",
        args.target_data[:-5],
        "\tPreprocess:",
        args.target_preprocess,
        "\tShape",
        list(target_data.shape),
    )

    return (
        source_dataset,
        source_dataloader_train,
        source_dataloader_eval,
        target_dataset,
        target_dataloader_train,
        target_dataloader_eval,
        gene_num,
        type_num,
        label_map,
        source_adata,
        target_adata,
    )


def adjacency(X, K=15):
    print("Computing KNN...")
    adj = kneighbors_graph(
        X.cpu().numpy(),
        K,
        mode="connectivity",
        include_self=True,
    ).toarray()
    adj = adj * adj.T
    return adj


def partition_data(
        predictions,
        prob_feature,
        prob_logit,
        source_dataset,
        target_dataset,
        args,
        iteration=0,
):
    """分区函数，使用自适应可靠性阈值"""

    # 计算综合可靠性
    combined_reliability = prob_feature * prob_logit

    # 自适应阈值策略
    if args.adaptive_threshold:
        # 根据迭代次数和可靠性分布动态调整阈值
        base_threshold = args.reliability_threshold

        # 计算当前可靠性的统计信息
        reliability_mean = combined_reliability.mean().item()
        reliability_std = combined_reliability.std().item()
        reliability_median = torch.median(combined_reliability).item()

        # 动态阈值计算：随着迭代进行，逐渐降低阈值以包含更多样本
        decay_factor = 0.95 ** iteration  # 每次迭代衰减5%
        adaptive_factor = min(reliability_mean + 0.5 * reliability_std, reliability_median + reliability_std)

        # 结合固定阈值和自适应因子
        dynamic_threshold = base_threshold * decay_factor + (1 - decay_factor) * adaptive_factor

        # 确保阈值在合理范围内
        dynamic_threshold = max(0.7, min(dynamic_threshold, 0.99))

        print(f"Adaptive threshold: {dynamic_threshold:.4f} (base: {base_threshold:.4f}, "
              f"mean_rel: {reliability_mean:.4f}, std_rel: {reliability_std:.4f})")

        # 使用动态阈值
        reliable_index = (prob_feature > dynamic_threshold) & (prob_logit > dynamic_threshold)
    else:
        reliable_index = (prob_feature > args.reliability_threshold) & (
                prob_logit > args.reliability_threshold
        )

    unreliable_index = ~reliable_index

    # 确保传播的标签具有足够的置信度
    if reliable_index.sum() > 0:
        reliable_predictions = predictions[reliable_index]
        reliable_probs = combined_reliability[reliable_index]

        # 移除置信度过低的异常预测
        confidence_threshold = torch.quantile(reliable_probs, 0.25).item()  # 保留置信度前75%的样本
        high_confidence_mask = reliable_probs >= confidence_threshold

        if high_confidence_mask.sum().item() < reliable_index.sum().item():
            print(
                f"Quality control: filtered {reliable_index.sum().item() - high_confidence_mask.sum().item()} low-confidence samples")

            # 更新可靠索引
            reliable_indices = torch.where(reliable_index)[0]
            filtered_indices = reliable_indices[high_confidence_mask]
            new_reliable_index = torch.zeros_like(reliable_index, dtype=torch.bool)
            new_reliable_index[filtered_indices] = True
            reliable_index = new_reliable_index
            unreliable_index = ~reliable_index

    # 合并可靠细胞到源数据集
    reliable_samples = target_dataset.tensors[0][reliable_index]
    reliable_predictions = predictions[reliable_index]
    source_data = torch.cat((source_dataset.tensors[0], reliable_samples))
    source_type = torch.cat(
        (source_dataset.tensors[1], reliable_predictions)
    )
    source_dataset = TensorDataset(source_data, source_type)

    # 保留不可靠细胞在目标数据集
    unreliable_samples = target_dataset.tensors[0][unreliable_index]
    unreliable_indices = target_dataset.tensors[1][unreliable_index]
    target_dataset = TensorDataSetWithIndex(unreliable_samples, unreliable_indices)

    print(
        "Source dataset size:",
        source_dataset.__len__(),
        "Target dataset size:",
        target_dataset.__len__(),
    )

    # 准备PyTorch DataLoader
    source_dataloader_train = DataLoader(
        dataset=source_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    source_dataloader_eval = DataLoader(
        dataset=source_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    target_dataloader_train = DataLoader(
        dataset=target_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    target_dataloader_eval = DataLoader(
        dataset=target_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    return (
        source_dataloader_train,
        source_dataloader_eval,
        target_dataloader_train,
        target_dataloader_eval,
        source_dataset,
        target_dataset,
    )