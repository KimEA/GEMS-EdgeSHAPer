"""
GC_GNN 단독 재학습 스크립트
==============================
trainer.py의 edge_weight 변경(Cα → Cβ) 후,
기존 GEMS 체크포인트를 유지한 채 GC_GNN만 seed 42로 재학습.

사용법:
  python retrain_gcn.py \
    --cleansplit_b6aepl /workspace/GEMS_pytorch_datasets/B6AEPL_train_cleansplit.pt \
    --pdbbind_b6aepl   /workspace/GEMS_pytorch_datasets/B6AEPL_train_pdbbind.pt \
    --split_json       results/pipeline/id_split.json \
    --output_dir       results/pipeline
"""

import os, sys, json, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.data_loader import (
    load_gems_dataset, get_dataset_info,
    apply_id_split, exclude_ids, load_id_split, create_dataloader,
)
from pipeline.trainer import (
    build_gcngnn, save_gcngnn_checkpoint, train_model,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleansplit_b6aepl", required=True)
    parser.add_argument("--pdbbind_b6aepl",   required=True)
    parser.add_argument("--split_json",        required=True)
    parser.add_argument("--output_dir",        default="results/pipeline")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--patience",type=int, default=50)
    parser.add_argument("--lr",      type=float, default=1e-4)
    parser.add_argument("--wd",      type=float, default=5e-4)
    parser.add_argument("--hidden",  type=int, default=256)
    parser.add_argument("--batch",   type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── 데이터 분할 로드
    train_ids, val_ids, test_ids = load_id_split(args.split_json)
    exclude_ids_set = val_ids | test_ids

    # ── CleanSplit 데이터
    print("\n[1] B6AEPL CleanSplit 로드...")
    b6aepl_clean = load_gems_dataset(args.cleansplit_b6aepl)
    info          = get_dataset_info(b6aepl_clean)
    train_cs, val_cs, _ = apply_id_split(b6aepl_clean, train_ids, val_ids, test_ids)

    # ── PDBbind 데이터
    print("\n[2] B6AEPL PDBbind 로드...")
    pdbbind_raw   = load_gems_dataset(args.pdbbind_b6aepl)
    pdbbind_clean = exclude_ids(pdbbind_raw, exclude_ids_set)

    train_config = {
        "lr":           args.lr,
        "weight_decay": args.wd,
        "optimizer":    "adam",
        "epochs":       args.epochs,
        "patience":     args.patience,
        "scheduler":    False,
    }

    def _train(train_data, val_data, name):
        torch.manual_seed(args.seed)
        model = build_gcngnn(info["node_feat_dim"], hidden=args.hidden, device=device)
        pin   = (device.type == "cuda")
        tl = create_dataloader(train_data, args.batch, shuffle=True,  pin_memory=pin)
        vl = create_dataloader(val_data,   args.batch, shuffle=False, pin_memory=pin)
        save_fn = lambda m, opt, ep, metrics, path: save_gcngnn_checkpoint(
            m, opt, ep, metrics, path,
            node_feat_dim=info["node_feat_dim"], hidden=args.hidden,
        )
        train_model(model, tl, vl, train_config, device,
                    save_dir=ckpt_dir, model_name=name, save_fn=save_fn)
        return model

    print(f"\n[3] GC_GNN_CleanSplit 재학습 (seed={args.seed}, edge_weight=Cβ)...")
    _train(train_cs, val_cs, f"gcn_cleansplit_seed{args.seed}")

    print(f"\n[4] GC_GNN_PDBbind 재학습 (seed={args.seed}, edge_weight=Cβ)...")
    _train(pdbbind_clean, val_cs, f"gcn_pdbbind_seed{args.seed}")

    print(f"\n완료. 체크포인트 저장 위치: {ckpt_dir}")
    print(f"  gcn_cleansplit_seed{args.seed}_best.pt")
    print(f"  gcn_pdbbind_seed{args.seed}_best.pt")


if __name__ == "__main__":
    main()
