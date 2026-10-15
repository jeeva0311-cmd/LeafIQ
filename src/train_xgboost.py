"""
LeafIQ - Phase 5: XGBoost Training
=====================================
Trains an XGBoost multiclass classifier on CNN embeddings extracted by
extract_features.py.  Uses validation features for early stopping and
hyperparameter selection.  Never touches test features during tuning.

Tuning strategy
---------------
1. Grid search over a small, focused set of hyperparameter combinations
   (learning_rate x max_depth x min_child_weight) using validation logloss
   as the selection criterion.
2. Train final model on train+val combined with the best hyperparameters
   (standard practice once hparams are fixed).
3. Save:
     models/xgboost_best.json          — XGBoost native format (portable)
     models/xgboost_best.pkl           — pickle (fast loading)
     outputs/xgboost_config.json       — hyperparameters + metadata
     outputs/xgboost_val_metrics.csv   — per-config validation scores
     outputs/xgboost_feature_importance.npy  — feature gain scores

Usage:
    python src/train_xgboost.py
    python src/train_xgboost.py --force   # retrain even if model exists
"""

import os, sys, json, csv, time, argparse, logging, warnings, pickle
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEAT_DIR     = PROJECT_ROOT / "outputs" / "features"
MODELS_DIR   = PROJECT_ROOT / "models"
LOG_DIR      = PROJECT_ROOT / "outputs" / "logs"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "train_xgboost.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

SEED = 42

CLASS_NAMES = [
    "Tomato_Healthy",
    "Tomato_Early_Blight",
    "Tomato_Late_Blight",
    "Tomato_Leaf_Mold",
]

# ── hyperparameter grid ───────────────────────────────────────────────────
# Small grid: keeps total configs ≤ 18, each takes <30 s on CPU
PARAM_GRID = {
    "learning_rate":    [0.05, 0.1, 0.2],
    "max_depth":        [4, 6, 8],
    "min_child_weight": [1, 3],
    # fixed across all configs
    "n_estimators":     500,          # capped by early stopping
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "early_stopping_rounds": 30,
}


# ── data loading ──────────────────────────────────────────────────────────

def load_features():
    splits = {}
    for s in ["train", "val", "test"]:
        fp = FEAT_DIR / f"{s}_features.npy"
        lp = FEAT_DIR / f"{s}_labels.npy"
        if not fp.exists():
            raise FileNotFoundError(
                f"Feature file not found: {fp}\n"
                "Run extract_features.py first."
            )
        X = np.load(fp)
        y = np.load(lp)
        splits[s] = (X, y)
        log.info("Loaded %s: X=%s  y=%s", s, X.shape, y.shape)
    return splits


# ── grid search ───────────────────────────────────────────────────────────

def grid_search(X_tr, y_tr, X_val, y_val) -> tuple[dict, list[dict]]:
    import xgboost as xgb

    fixed = {
        "objective":        "multi:softprob",
        "num_class":        len(CLASS_NAMES),
        "eval_metric":      "mlogloss",
        "subsample":        PARAM_GRID["subsample"],
        "colsample_bytree": PARAM_GRID["colsample_bytree"],
        "gamma":            PARAM_GRID["gamma"],
        "reg_alpha":        PARAM_GRID["reg_alpha"],
        "reg_lambda":       PARAM_GRID["reg_lambda"],
        "seed":             SEED,
        "nthread":          -1,
        "tree_method":      "hist",   # fast histogram method
        "verbosity":        0,
    }

    dtrain = xgb.DMatrix(X_tr,  label=y_tr)
    dval   = xgb.DMatrix(X_val, label=y_val)

    configs = []
    for lr in PARAM_GRID["learning_rate"]:
        for depth in PARAM_GRID["max_depth"]:
            for mcw in PARAM_GRID["min_child_weight"]:
                configs.append({"learning_rate": lr,
                                 "max_depth":    depth,
                                 "min_child_weight": mcw})

    log.info("Grid search: %d configurations ...", len(configs))
    results = []
    best_val_loss = float("inf")
    best_params   = None

    for i, cfg in enumerate(configs):
        params = {**fixed, **cfg,
                  "learning_rate": cfg["learning_rate"],
                  "max_depth":     cfg["max_depth"],
                  "min_child_weight": cfg["min_child_weight"]}

        t0 = time.perf_counter()
        evals_result = {}
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=PARAM_GRID["n_estimators"],
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=PARAM_GRID["early_stopping_rounds"],
            evals_result=evals_result,
            verbose_eval=False,
        )
        elapsed = time.perf_counter() - t0

        best_iter = model.best_iteration
        val_loss  = evals_result["val"]["mlogloss"][best_iter]
        tr_loss   = evals_result["train"]["mlogloss"][best_iter]

        # val accuracy
        val_probs = model.predict(dval).reshape(-1, len(CLASS_NAMES))
        val_preds = val_probs.argmax(axis=1)
        val_acc   = float((val_preds == y_val).mean())

        row = {
            "config_id":        i + 1,
            "learning_rate":    cfg["learning_rate"],
            "max_depth":        cfg["max_depth"],
            "min_child_weight": cfg["min_child_weight"],
            "best_round":       best_iter,
            "val_logloss":      round(val_loss, 6),
            "train_logloss":    round(tr_loss, 6),
            "val_accuracy":     round(val_acc, 6),
            "elapsed_s":        round(elapsed, 2),
        }
        results.append(row)
        log.info(
            "  [%2d/%2d] lr=%.2f depth=%d mcw=%d  "
            "rounds=%d  val_loss=%.4f  val_acc=%.4f  (%.1fs)",
            i+1, len(configs),
            cfg["learning_rate"], cfg["max_depth"], cfg["min_child_weight"],
            best_iter, val_loss, val_acc, elapsed
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params   = {**cfg}
            best_params["best_round"] = best_iter

    log.info("Best config: %s  val_logloss=%.6f", best_params, best_val_loss)
    return best_params, results


# ── final training ────────────────────────────────────────────────────────

def train_final(best_params: dict, X_tr, y_tr, X_val, y_val):
    """
    Train on train+val combined using best hyperparameters.
    Use best_round + 10% headroom to avoid underfitting.
    """
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, f1_score

    X_all = np.concatenate([X_tr, X_val], axis=0)
    y_all = np.concatenate([y_tr, y_val], axis=0)
    log.info("Final training on train+val: %d samples", len(X_all))

    n_rounds = int(best_params["best_round"] * 1.1) + 10
    log.info("  n_rounds = %d", n_rounds)

    params = {
        "objective":        "multi:softprob",
        "num_class":        len(CLASS_NAMES),
        "eval_metric":      "mlogloss",
        "learning_rate":    best_params["learning_rate"],
        "max_depth":        best_params["max_depth"],
        "min_child_weight": best_params["min_child_weight"],
        "subsample":        PARAM_GRID["subsample"],
        "colsample_bytree": PARAM_GRID["colsample_bytree"],
        "gamma":            PARAM_GRID["gamma"],
        "reg_alpha":        PARAM_GRID["reg_alpha"],
        "reg_lambda":       PARAM_GRID["reg_lambda"],
        "seed":             SEED,
        "nthread":          -1,
        "tree_method":      "hist",
        "verbosity":        0,
    }

    dall  = xgb.DMatrix(X_all, label=y_all)
    dval  = xgb.DMatrix(X_val, label=y_val)

    t0 = time.perf_counter()
    model = xgb.train(params, dall, num_boost_round=n_rounds, verbose_eval=False)
    elapsed = time.perf_counter() - t0
    log.info("  Training done in %.1fs", elapsed)

    # Quick val check (data was used for hparam selection, not for final fit)
    val_probs = model.predict(dval).reshape(-1, len(CLASS_NAMES))
    val_preds = val_probs.argmax(axis=1)
    val_acc   = accuracy_score(y_val, val_preds)
    val_f1    = f1_score(y_val, val_preds, average="macro", zero_division=0)
    log.info("  Val accuracy=%.4f  Val macro-F1=%.4f", val_acc, val_f1)

    return model, params, n_rounds, val_acc, val_f1


# ── save ─────────────────────────────────────────────────────────────────

def save_model(model, params: dict, best_params: dict, n_rounds: int,
               val_acc: float, val_f1: float,
               grid_results: list[dict]) -> dict:
    import xgboost as xgb

    # Native XGBoost format
    json_path = MODELS_DIR / "xgboost_best.json"
    model.save_model(str(json_path))
    log.info("Saved model (JSON): %s", json_path)

    # Pickle for fast loading
    pkl_path = MODELS_DIR / "xgboost_best.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    log.info("Saved model (pkl): %s", pkl_path)

    # Config
    config = {
        "model_key":       "cnn_xgboost",
        "display_name":    "CNN + XGBoost",
        "cnn_checkpoint":  "models/custom_cnn_best.keras",
        "feature_dir":     "outputs/features",
        "embedding_dim":   256,
        "class_names":     CLASS_NAMES,
        "n_classes":       len(CLASS_NAMES),
        "seed":            SEED,
        "xgboost_params":  params,
        "best_hparams":    best_params,
        "n_rounds":        n_rounds,
        "val_accuracy":    round(val_acc, 6),
        "val_f1_macro":    round(val_f1, 6),
    }
    cfg_path = OUTPUTS_DIR / "xgboost_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    log.info("Config saved: %s", cfg_path)

    # Grid search results CSV
    grid_path = OUTPUTS_DIR / "xgboost_val_metrics.csv"
    if grid_results:
        fields = list(grid_results[0].keys())
        with open(grid_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(grid_results)
        log.info("Grid results: %s", grid_path)

    # Feature importance
    scores = model.get_score(importance_type="gain")
    # Map f0..fN to importance array
    n_feat = 256
    fi = np.zeros(n_feat, dtype=np.float32)
    for k, v in scores.items():
        idx = int(k.lstrip("f"))
        if idx < n_feat:
            fi[idx] = float(v)
    fi_path = OUTPUTS_DIR / "xgboost_feature_importance.npy"
    np.save(fi_path, fi)
    log.info("Feature importance: %s", fi_path)

    return config


# ── main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LeafIQ Phase 5 XGBoost Training")
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if model already exists")
    parser.add_argument("--skip-grid", action="store_true",
                        help="Skip grid search, use default best params")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("LeafIQ -- Phase 5: XGBoost Training")
    log.info("=" * 60)

    pkl_path = MODELS_DIR / "xgboost_best.pkl"
    if pkl_path.exists() and not args.force:
        log.info("XGBoost model already exists. Use --force to retrain.")
        log.info("  %s", pkl_path)
        # Print saved config
        cfg_path = OUTPUTS_DIR / "xgboost_config.json"
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            log.info("  val_accuracy=%.4f  val_f1=%.4f",
                     cfg.get("val_accuracy", 0), cfg.get("val_f1_macro", 0))
        return

    splits = load_features()
    X_tr, y_tr   = splits["train"]
    X_val, y_val = splits["val"]
    X_te, y_te   = splits["test"]   # loaded but NOT used for tuning

    log.info("Train: %d  Val: %d  Test: %d (not used for tuning)",
             len(X_tr), len(X_val), len(X_te))

    if args.skip_grid:
        log.info("Skipping grid search — using default params.")
        best_params = {
            "learning_rate": 0.1,
            "max_depth":     6,
            "min_child_weight": 1,
            "best_round": 100,
        }
        grid_results = []
    else:
        best_params, grid_results = grid_search(X_tr, y_tr, X_val, y_val)

    model, params, n_rounds, val_acc, val_f1 = train_final(
        best_params, X_tr, y_tr, X_val, y_val
    )

    config = save_model(model, params, best_params, n_rounds,
                        val_acc, val_f1, grid_results)

    log.info("\nXGBoost training complete.")
    log.info("  Model     : %s", MODELS_DIR / "xgboost_best.json")
    log.info("  Val acc   : %.4f", val_acc)
    log.info("  Val F1    : %.4f", val_f1)
    log.info("  Config    : %s", OUTPUTS_DIR / "xgboost_config.json")


if __name__ == "__main__":
    main()
