"""
holdout 샘플의 엣지 거리 정보를 Shapley CSV에 추가하는 스크립트.

edge_attr 거리 컬럼 (interaction 엣지 기준):
  index 3: Cα 거리 / 10
  index 4: N  거리 / 10
  index 5: C  거리 / 10
  index 6: Cβ 거리 / 10  ← 사이드체인 시작점, 화학적 접촉에 가장 가까운 프록시

distance_A    = edge_attr[:, 6] * 10  (Cβ 거리, Å)
distance_A_ca = edge_attr[:, 3] * 10  (Cα 거리, Å, 비교용 보존)

사용법:
  # seed42 결과
  python analysis/extract_edge_distances.py \
    --xai_dir    results/pipeline/xai/holdout \
    --split_json results/pipeline/id_split.json \
    --dataset    /workspace/GEMS_pytorch_datasets/B6AEPL_train_cleansplit.pt

  # seeds 123+2024 결과
  python analysis/extract_edge_distances.py \
    --xai_dir    results/pipeline_seeds/xai/holdout \
    --split_json results/pipeline/id_split.json \
    --dataset    /workspace/GEMS_pytorch_datasets/B6AEPL_train_cleansplit.pt
"""
import os, json, argparse, torch
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DIST_COL_CA = 3
DIST_COL_CB = 6
DIST_SCALE  = 10


def load_dataset(path):
    print(f"[load] {path} ...")
    data   = torch.load(path, map_location="cpu")
    graphs = list(data) if hasattr(data, "__iter__") else [data[i] for i in range(len(data))]
    print(f"  {len(graphs)} 샘플 로드 완료")
    return graphs


def build_id_map(graphs):
    return {g.id: g for g in graphs}


def add_distance_to_csv(csv_path, graph):
    df        = pd.read_csv(csv_path)
    edge_attr = graph.edge_attr
    n_cols    = edge_attr.shape[1] if edge_attr is not None else 0

    if "distance_A" in df.columns and "distance_A_ca" not in df.columns:
        df = df.rename(columns={"distance_A": "distance_A_ca"})

    def extract(col_idx):
        if edge_attr is None or n_cols <= col_idx:
            return df["edge_idx"].apply(lambda _: float("nan"))
        raw = edge_attr[:, col_idx].numpy() * DIST_SCALE
        return df["edge_idx"].apply(
            lambda i: float(raw[i]) if i < len(raw) else float("nan")
        )

    if "distance_A_ca" not in df.columns:
        df["distance_A_ca"] = extract(DIST_COL_CA)
    df["distance_A"] = extract(DIST_COL_CB)
    df.to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xai_dir",
        default=os.path.join(_ROOT, "results/pipeline/xai/holdout"),
        help="XAI holdout 결과 디렉터리 (seed*/model/group/pdb_id/ 구조)")
    parser.add_argument("--split_json",
        default=os.path.join(_ROOT, "results/pipeline/id_split.json"))
    parser.add_argument("--dataset",
        default="/workspace/GEMS_pytorch_datasets/B6AEPL_train_cleansplit.pt")
    args = parser.parse_args()

    with open(args.split_json) as f:
        split = json.load(f)
    print(f"[split] holdout 샘플 수: {len(split.get('test', []))}")

    graphs = load_dataset(args.dataset)
    id_map = build_id_map(graphs)

    processed, skipped = 0, 0
    for seed_name in sorted(os.listdir(args.xai_dir)):
        seed_path = os.path.join(args.xai_dir, seed_name)
        if not os.path.isdir(seed_path):
            continue
        for model_name in sorted(os.listdir(seed_path)):
            model_path = os.path.join(seed_path, model_name)
            if not os.path.isdir(model_path):
                continue
            for grp in ["low", "medium", "high"]:
                grp_path = os.path.join(model_path, grp)
                if not os.path.isdir(grp_path):
                    continue
                for pdb_id in sorted(os.listdir(grp_path)):
                    csv_path = os.path.join(grp_path, pdb_id, f"{pdb_id}_shapley.csv")
                    if not os.path.exists(csv_path):
                        continue
                    if pdb_id not in id_map:
                        skipped += 1
                        continue
                    add_distance_to_csv(csv_path, id_map[pdb_id])
                    processed += 1

    print(f"\n완료: {processed}개 CSV 업데이트, {skipped}개 스킵")


if __name__ == "__main__":
    main()
