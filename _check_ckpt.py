import sys, torch
path = sys.argv[1] if len(sys.argv) > 1 else "training/candi_polar_x0_corr_best.pt"
c = torch.load(path, map_location="cpu", weights_only=False)
for k in ["epoch", "val_cont", "val_disc", "pred_type", "polar"]:
    print(f"{k}: {c.get(k, '?')}")
