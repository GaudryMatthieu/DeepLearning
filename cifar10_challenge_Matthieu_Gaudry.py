import argparse
import gc
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # moins de logs TF
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # pas besoin d'écran : toutes les figures sont sauvegardées
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
NUM_CLASSES = 10
AUTOTUNE = tf.data.AUTOTUNE


# =============================================================================
# 1. CONFIGURATION
# =============================================================================
@dataclass
class TrainConfig:
    model_name: str = "resnet9"
    epochs: int = 30
    batch_size: int = 128
    max_lr: float = 0.1               
    momentum: float = 0.9
    weight_decay: float = 5e-4        
    label_smoothing: float = 0.1
    augment: str = "full"             
    warmup_pct: float = 0.2           
    tag: str = ""                     


# =============================================================================
# 2. DONNÉES
# =============================================================================
class CIFAR10Data:
    def __init__(self, val_size: int = 5000, seed: int = 42):
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        y_train, y_test = y_train.flatten(), y_test.flatten()

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(x_train))
        val_idx, tr_idx = idx[:val_size], idx[val_size:]

        self.x_train, self.y_train = x_train[tr_idx], y_train[tr_idx]
        self.x_val, self.y_val = x_train[val_idx], y_train[val_idx]
        self.x_test, self.y_test = x_test, y_test

        x = self.x_train.astype(np.float32) / 255.0
        self.mean = x.mean(axis=(0, 1, 2)).astype(np.float32)
        self.std = x.std(axis=(0, 1, 2)).astype(np.float32)

        print(f"[DATA] train={len(self.x_train)}  val={len(self.x_val)}  test={len(self.x_test)}")
        print(f"[DATA] mean={np.round(self.mean, 4)}  std={np.round(self.std, 4)}")

    def normalize_np(self, x: np.ndarray) -> np.ndarray:
        return ((x.astype(np.float32) / 255.0) - self.mean) / self.std

    def subset(self, n: int, seed: int = 0):
        if n is None or n >= len(self.x_train):
            return self.x_train, self.y_train
        rng = np.random.default_rng(seed)
        per_class = n // NUM_CLASSES
        idx = np.concatenate([
            rng.choice(np.where(self.y_train == c)[0], per_class, replace=False)
            for c in range(NUM_CLASSES)
        ])
        rng.shuffle(idx)
        return self.x_train[idx], self.y_train[idx]

    @staticmethod
    def _cutout(img: tf.Tensor, size: int = 8) -> tf.Tensor:
        h, w = 32, 32
        cy = tf.random.uniform([], 0, h, dtype=tf.int32)
        cx = tf.random.uniform([], 0, w, dtype=tf.int32)
        y1, y2 = tf.clip_by_value(cy - size // 2, 0, h), tf.clip_by_value(cy + size // 2, 0, h)
        x1, x2 = tf.clip_by_value(cx - size // 2, 0, w), tf.clip_by_value(cx + size // 2, 0, w)
        rows = tf.range(h)[:, None]
        cols = tf.range(w)[None, :]
        inside = (rows >= y1) & (rows < y2) & (cols >= x1) & (cols < x2)
        mask = tf.cast(tf.logical_not(inside), img.dtype)[..., None]
        return img * mask  # 0 = valeur moyenne après normalisation

    def _augment(self, img: tf.Tensor, level: str) -> tf.Tensor:
        if level in ("basic", "full"):
            img = tf.pad(img, [[4, 4], [4, 4], [0, 0]], mode="REFLECT")
            img = tf.image.random_crop(img, [32, 32, 3])
            img = tf.image.random_flip_left_right(img)
        if level == "full":
            img = self._cutout(img, 8)
        return img

    def make_dataset(self, x, y, batch_size: int, training: bool, augment: str = "none"):
        mean, std = tf.constant(self.mean), tf.constant(self.std)

        def prep(img, label):
            img = (tf.cast(img, tf.float32) / 255.0 - mean) / std
            if training and augment != "none":
                img = self._augment(img, augment)
            return img, tf.one_hot(label, NUM_CLASSES)

        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if training:
            ds = ds.shuffle(len(x), reshuffle_each_iteration=True)
        return ds.map(prep, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)


# =============================================================================
# 3. MODÈLES
# =============================================================================
class ModelZoo:
    @staticmethod
    def _conv_bn_relu(x, filters, wd):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                          kernel_initializer="he_normal",
                          kernel_regularizer=regularizers.l2(wd))(x)
        x = layers.BatchNormalization()(x)
        return layers.Activation("relu")(x)

    @staticmethod
    def _head(x, wd):
        x = layers.Dense(NUM_CLASSES, kernel_regularizer=regularizers.l2(wd))(x)
        # float32 en sortie pour la stabilité en mixed precision
        return layers.Activation("softmax", dtype="float32")(x)


    @classmethod
    def home_made(cls, wd: float) -> keras.Model:
        inp = keras.Input((32, 32, 3))
        
        x = layers.Conv2D(75, (3, 3), strides=1, padding="same", activation="relu", 
                          kernel_regularizer=regularizers.l2(wd))(inp)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), strides=2, padding="same")(x)
        
        x = layers.Conv2D(50, (3, 3), strides=1, padding="same", activation="relu", 
                          kernel_regularizer=regularizers.l2(wd))(x)
        x = layers.Dropout(0.2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), strides=2, padding="same")(x)
        
        x = layers.Conv2D(25, (3, 3), strides=1, padding="same", activation="relu", 
                          kernel_regularizer=regularizers.l2(wd))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), strides=2, padding="same")(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(wd))(x)
        x = layers.Dropout(0.3)(x)
        
        return keras.Model(inp, cls._head(x, wd), name="home_made")

    @classmethod
    def mlp(cls, wd: float) -> keras.Model:
        inp = keras.Input((32, 32, 3))
        x = layers.Flatten()(inp)
        for units in (1024, 512):
            x = layers.Dense(units, kernel_regularizer=regularizers.l2(wd))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.Dropout(0.3)(x)
        return keras.Model(inp, cls._head(x, wd), name="mlp")

    @classmethod
    def simple_cnn(cls, wd: float) -> keras.Model:
        inp = keras.Input((32, 32, 3))
        x = inp
        for f in (32, 64):
            x = layers.Conv2D(f, 3, padding="same", activation="relu",
                              kernel_regularizer=regularizers.l2(wd))(x)
            x = layers.Conv2D(f, 3, padding="same", activation="relu",
                              kernel_regularizer=regularizers.l2(wd))(x)
            x = layers.MaxPooling2D()(x)
            x = layers.Dropout(0.25)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(wd))(x)
        x = layers.Dropout(0.5)(x)
        return keras.Model(inp, cls._head(x, wd), name="simple_cnn")

    @classmethod
    def vgg_bn(cls, wd: float) -> keras.Model:
        inp = keras.Input((32, 32, 3))
        x = inp
        for f, drop in ((64, 0.2), (128, 0.3), (256, 0.4)):
            x = cls._conv_bn_relu(x, f, wd)
            x = cls._conv_bn_relu(x, f, wd)
            x = layers.MaxPooling2D()(x)
            x = layers.Dropout(drop)(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, use_bias=False, kernel_regularizer=regularizers.l2(wd))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.5)(x)
        return keras.Model(inp, cls._head(x, wd), name="vgg_bn")

    @classmethod
    def resnet9(cls, wd: float) -> keras.Model:
        cbr = lambda t, f: cls._conv_bn_relu(t, f, wd)
        inp = keras.Input((32, 32, 3))
        x = cbr(inp, 64)                                  
        x = cbr(x, 128)
        x = layers.MaxPooling2D()(x)                      
        r = cbr(cbr(x, 128), 128)
        x = layers.Add()([x, r])                          
        x = cbr(x, 256)
        x = layers.MaxPooling2D()(x)                  
        x = cbr(x, 512)
        x = layers.MaxPooling2D()(x)                 
        r = cbr(cbr(x, 512), 512)
        x = layers.Add()([x, r])                      
        x = layers.GlobalMaxPooling2D()(x)
        x = layers.Dropout(0.2)(x)
        return keras.Model(inp, cls._head(x, wd), name="resnet9")

    REGISTRY: Dict[str, Dict] = {
        "mlp":        {"max_lr": 0.05, "augment": "none"},
        "simple_cnn": {"max_lr": 0.05, "augment": "basic"},
        "vgg_bn":     {"max_lr": 0.1,  "augment": "full"},
        "resnet9":    {"max_lr": 0.1,  "augment": "full"},
        "home_made":  {"max_lr": 0.05, "augment": "basic"}, 
    }

    @classmethod
    def build(cls, name: str, wd: float) -> keras.Model:
        builder: Callable = getattr(cls, name, None)
        if builder is None or name not in cls.REGISTRY:
            raise ValueError(f"Modèle inconnu : {name}. Choix : {list(cls.REGISTRY)}")
        return builder(wd)


# =============================================================================
# 4. PLANNING DU LEARNING RATE (One-Cycle : warmup linéaire + décroissance cosinus)
# =============================================================================
class OneCycleSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr: float, total_steps: int, warmup_pct: float = 0.2,
                 div_start: float = 25.0, div_end: float = 1e4):
        super().__init__()
        self.max_lr = float(max_lr)
        self.total_steps = float(max(1, total_steps))
        self.warmup_steps = float(max(1, int(total_steps * warmup_pct)))
        self.start_lr = self.max_lr / div_start
        self.end_lr = self.max_lr / div_end

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        lr_warm = self.start_lr + (self.max_lr - self.start_lr) * step / self.warmup_steps
        progress = (step - self.warmup_steps) / tf.maximum(1.0, self.total_steps - self.warmup_steps)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        lr_cos = self.end_lr + 0.5 * (self.max_lr - self.end_lr) * (1.0 + tf.cos(math.pi * progress))
        return tf.where(step < self.warmup_steps, lr_warm, lr_cos)

    def get_config(self):
        return {"max_lr": self.max_lr, "total_steps": self.total_steps,
                "warmup_steps": self.warmup_steps}


# =============================================================================
# 5. ENTRAÎNEMENT
# =============================================================================
class Trainer:
    def __init__(self, data: CIFAR10Data, out_dir: str):
        self.data = data
        self.out_dir = out_dir

    def train(self, cfg: TrainConfig, x_train, y_train) -> Dict:
        keras.backend.clear_session()
        gc.collect()

        run_name = f"{cfg.tag}_{cfg.model_name}".strip("_")
        print("\n" + "=" * 78)
        print(f"[TRAIN] {run_name} | n_train={len(x_train)} | {asdict(cfg)}")
        print("=" * 78)

        train_ds = self.data.make_dataset(x_train, y_train, cfg.batch_size, True, cfg.augment)
        val_ds = self.data.make_dataset(self.data.x_val, self.data.y_val, 512, False)

        model = ModelZoo.build(cfg.model_name, cfg.weight_decay)
        steps = math.ceil(len(x_train) / cfg.batch_size) * cfg.epochs
        optimizer = keras.optimizers.SGD(
            learning_rate=OneCycleSchedule(cfg.max_lr, steps, cfg.warmup_pct),
            momentum=cfg.momentum, nesterov=True)
        model.compile(optimizer=optimizer,
                      loss=keras.losses.CategoricalCrossentropy(label_smoothing=cfg.label_smoothing),
                      metrics=["accuracy"])

        ckpt_path = os.path.join(self.out_dir, "weights", f"{run_name}.weights.h5")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        callbacks = [
            keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", mode="max",
                                            save_best_only=True, save_weights_only=True, verbose=0),
            keras.callbacks.CSVLogger(os.path.join(self.out_dir, "logs", f"{run_name}.csv")),
        ]
        os.makedirs(os.path.join(self.out_dir, "logs"), exist_ok=True)

        t0 = time.time()
        hist = model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs,
                         callbacks=callbacks, verbose=2)
        train_time = time.time() - t0
        model.load_weights(ckpt_path)  # meilleur epoch sur la validation

        return {"model": model, "history": hist.history, "train_time": train_time,
                "run_name": run_name, "params": model.count_params()}


# =============================================================================
# 6. ÉVALUATION & VISUALISATIONS
# =============================================================================
class Evaluator:
    def __init__(self, data: CIFAR10Data, out_dir: str):
        self.data = data
        self.out_dir = out_dir
        self.fig_dir = os.path.join(out_dir, "figures")
        os.makedirs(self.fig_dir, exist_ok=True)
        self.x_test_norm = data.normalize_np(data.x_test)

    def predict(self, model, tta: bool = False) -> np.ndarray:
        probs = model.predict(self.x_test_norm, batch_size=512, verbose=0)
        if tta:  # Test-Time Augmentation : moyenne image + image miroir
            probs = (probs + model.predict(self.x_test_norm[:, :, ::-1, :],
                                           batch_size=512, verbose=0)) / 2.0
        return probs

    def evaluate(self, model, run_name: str, detailed: bool = False) -> Dict:
        y = self.data.y_test
        probs = self.predict(model, tta=False)
        acc = float((probs.argmax(1) == y).mean())
        probs_tta = self.predict(model, tta=True)
        y_pred = probs_tta.argmax(1)
        acc_tta = float((y_pred == y).mean())
        top2 = float(np.mean([y[i] in np.argsort(probs_tta[i])[-2:] for i in range(len(y))]))
        print(f"[EVAL] {run_name}: test_acc={acc:.4f} | test_acc_TTA={acc_tta:.4f} | top-2={top2:.4f}")

        if detailed:
            report = classification_report(y, y_pred, target_names=CLASS_NAMES, digits=4)
            print(report)
            with open(os.path.join(self.out_dir, f"{run_name}_classification_report.txt"), "w") as f:
                f.write(report)
            self.plot_confusion(y, y_pred, run_name)
            self.plot_misclassified(y, y_pred, probs_tta, run_name)
        return {"test_acc": acc, "test_acc_tta": acc_tta, "test_top2": top2}

    def plot_history(self, history: Dict, run_name: str):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(history["loss"], label="train")
        ax[0].plot(history["val_loss"], label="validation")
        ax[0].set_title(f"Loss - {run_name}"); ax[0].set_xlabel("epoch"); ax[0].legend()
        ax[1].plot(history["accuracy"], label="train")
        ax[1].plot(history["val_accuracy"], label="validation")
        ax[1].set_title(f"Accuracy - {run_name}"); ax[1].set_xlabel("epoch"); ax[1].legend()
        for a in ax: a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.fig_dir, f"{run_name}_history.png"), dpi=120)
        plt.close(fig)

    def plot_confusion(self, y, y_pred, run_name: str):
        cm = confusion_matrix(y, y_pred, normalize="true")
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_NAMES)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 0.5 else "black")
        ax.set_xlabel("Prédit"); ax.set_ylabel("Réel"); ax.set_title(f"Matrice de confusion - {run_name}")
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(os.path.join(self.fig_dir, f"{run_name}_confusion.png"), dpi=120)
        plt.close(fig)

    def plot_misclassified(self, y, y_pred, probs, run_name: str, n: int = 25):
        wrong = np.where(y != y_pred)[0]
        if len(wrong) == 0:
            return
        wrong = wrong[np.argsort(-probs[wrong].max(1))][:n]
        fig, axes = plt.subplots(5, 5, figsize=(10, 11))
        for ax, i in zip(axes.flat, wrong):
            ax.imshow(self.data.x_test[i])
            ax.set_title(f"réel: {CLASS_NAMES[y[i]]}\nprédit: {CLASS_NAMES[y_pred[i]]} ({probs[i].max():.2f})",
                         fontsize=8)
            ax.axis("off")
        fig.suptitle(f"Erreurs les plus confiantes - {run_name}")
        fig.tight_layout()
        fig.savefig(os.path.join(self.fig_dir, f"{run_name}_misclassified.png"), dpi=120)
        plt.close(fig)

    def plot_comparison(self, df: pd.DataFrame, title: str, fname: str, label_col: str = "run"):
        fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(df)), 4.5))
        bars = ax.bar(df[label_col], df["test_acc_tta"] * 100, color="#4C72B0")
        for b, v in zip(bars, df["test_acc_tta"] * 100):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
        ax.set_ylabel("Test accuracy (TTA) %"); ax.set_title(title)
        ax.set_ylim(0, 100); ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=20, ha="right"); fig.tight_layout()
        fig.savefig(os.path.join(self.fig_dir, fname), dpi=120)
        plt.close(fig)

    def plot_val_curves(self, runs: List[Dict], title: str, fname: str):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for r in runs:
            ax.plot(r["history"]["val_accuracy"], label=r["run"])
        ax.set_xlabel("epoch"); ax.set_ylabel("val accuracy"); ax.set_title(title)
        ax.grid(alpha=0.3); ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(os.path.join(self.fig_dir, fname), dpi=120)
        plt.close(fig)


# =============================================================================
# 7. ORCHESTRATION DES EXPÉRIENCES
# =============================================================================
class Experiment:
    def __init__(self, args):
        self.args = args
        self.out_dir = args.out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.data = CIFAR10Data(val_size=5000, seed=args.seed)
        self.trainer = Trainer(self.data, self.out_dir)
        self.evaluator = Evaluator(self.data, self.out_dir)
        self.all_results: List[Dict] = []

    def _run(self, cfg: TrainConfig, x, y, detailed: bool = False, keep_history=True) -> Dict:
        out = self.trainer.train(cfg, x, y)
        metrics = self.evaluator.evaluate(out["model"], out["run_name"], detailed=detailed)
        self.evaluator.plot_history(out["history"], out["run_name"])
        res = {
            "run": out["run_name"], "phase": cfg.tag, "model": cfg.model_name,
            "n_train": len(x), "epochs": cfg.epochs, "max_lr": cfg.max_lr,
            "augment": cfg.augment, "label_smoothing": cfg.label_smoothing,
            "params": out["params"], "train_time_s": round(out["train_time"], 1),
            "best_val_acc": float(max(out["history"]["val_accuracy"])),
            "final_train_acc": float(out["history"]["accuracy"][-1]),
            **metrics,
        }
        res["overfit_gap"] = res["final_train_acc"] - res["best_val_acc"]
        if keep_history:
            res["history"] = {k: [float(v) for v in vals] for k, vals in out["history"].items()}
        self.all_results.append(res)
        self._save_results()
        del out
        return res

    def _cfg(self, model_name: str, epochs: int, tag: str, **overrides) -> TrainConfig:
        params = dict(ModelZoo.REGISTRY[model_name])
        params.update(overrides)
        return TrainConfig(model_name=model_name, epochs=epochs, tag=tag,
                           batch_size=self.args.batch_size, **params)

    def _save_results(self):
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "history"} for r in self.all_results])
        df.to_csv(os.path.join(self.out_dir, "results_summary.csv"), index=False)
        with open(os.path.join(self.out_dir, "results_full.json"), "w") as f:
            json.dump(self.all_results, f, indent=2)

    @staticmethod
    def _print_table(runs: List[Dict], title: str):
        cols = ["run", "n_train", "epochs", "params", "train_time_s",
                "best_val_acc", "test_acc", "test_acc_tta", "overfit_gap"]
        df = pd.DataFrame(runs)[cols]
        print("\n" + "#" * 78 + f"\n# {title}\n" + "#" * 78)
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        return df

    def benchmark(self) -> str:
        a = self.args
        x, y = self.data.subset(a.subset_size, seed=a.seed)
        runs = [self._run(self._cfg(m, a.bench_epochs, "bench"), x, y) for m in a.bench_models]
        df = self._print_table(runs, f"PHASE 1 - BENCHMARK ({len(x)} images, {a.bench_epochs} epochs)")
        self.evaluator.plot_comparison(df, "Benchmark des modèles (sous-ensemble)", "bench_comparison.png")
        self.evaluator.plot_val_curves(runs, "Benchmark - accuracy validation", "bench_val_curves.png")
        best = max(runs, key=lambda r: r["best_val_acc"])["model"]  # sélection sur la VALIDATION
        print(f"\n>>> Meilleur modèle du benchmark (val) : {best}")
        return best

    def tune(self, model_name: str) -> Dict:
        a = self.args
        x, y = self.data.subset(a.subset_size, seed=a.seed)
        base = ModelZoo.REGISTRY[model_name]
        lr_runs = [self._run(self._cfg(model_name, a.bench_epochs, f"tune_lr{lr}", max_lr=lr), x, y)
                   for lr in a.lr_grid]
        best_lr = max(lr_runs, key=lambda r: r["best_val_acc"])["max_lr"]
        self._print_table(lr_runs, f"PHASE 2a - RECHERCHE DU LEARNING RATE ({model_name})")
        variants = [("tune_noaug", {"augment": "none"}),
                    ("tune_basicaug", {"augment": "basic"}),
                    ("tune_fullaug", {"augment": "full"}),
                    ("tune_fullaug_noLS", {"augment": "full", "label_smoothing": 0.0})]
        abl_runs = [self._run(self._cfg(model_name, a.bench_epochs, tag, max_lr=best_lr, **ov), x, y)
                    for tag, ov in variants]
        df = self._print_table(abl_runs, f"PHASE 2b - ABLATION AUGMENTATION / LABEL SMOOTHING ({model_name})")
        self.evaluator.plot_comparison(df, f"Ablation - {model_name}", "tune_ablation.png")
        self.evaluator.plot_val_curves(lr_runs + abl_runs, f"Tuning - {model_name}", "tune_val_curves.png")
        best = max(abl_runs, key=lambda r: r["best_val_acc"])
        best_params = {"max_lr": best_lr, "augment": best["augment"],
                       "label_smoothing": best["label_smoothing"]}
        print(f"\n>>> Meilleurs hyperparamètres retenus : {best_params}")
        return best_params

    def final(self, models: List[str], tuned: Optional[Dict] = None):
        a = self.args
        runs = []
        for m in models:
            overrides = dict(tuned) if (tuned and m == tuned.get("_model")) else {}
            overrides.pop("_model", None)
            cfg = self._cfg(m, a.final_epochs, "final", **overrides)
            runs.append(self._run(cfg, self.data.x_train, self.data.y_train, detailed=True))
        df = self._print_table(runs, f"PHASE 3 - DATASET COMPLET ({len(self.data.x_train)} images)")
        self.evaluator.plot_comparison(df, "Modèles finaux (dataset complet)", "final_comparison.png")
        best = max(runs, key=lambda r: r["test_acc_tta"])
        print(f"\n*** RÉSULTAT FINAL : {best['run']} -> test accuracy = {best['test_acc']*100:.2f}% "
              f"| avec TTA = {best['test_acc_tta']*100:.2f}% ***")

    def run(self):
        a = self.args
        if a.home_made:
            print("\n>>> Exécution exclusive du modèle HOME-MADE sur le dataset complet <<<")
            self.final(["home_made"])
            print(f"\nTous les résultats sont dans : {os.path.abspath(self.out_dir)}")
            return
        tuned = None
        if a.phase in ("benchmark", "all"):
            best = self.benchmark()
            if a.phase == "all" and not a.skip_tuning:
                tuned = self.tune(best)
                tuned["_model"] = best
        if a.phase == "tune":
            tuned = self.tune(a.tune_model)
            tuned["_model"] = a.tune_model
        if a.phase in ("final", "all"):
            self.final(a.final_models, tuned)
        print(f"\nTous les résultats sont dans : {os.path.abspath(self.out_dir)}")


# =============================================================================
# 8. POINT D'ENTRÉE
# =============================================================================
def setup_hardware(use_mixed_precision: bool):
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    if gpus:
        print(f"[HW] {len(gpus)} GPU détecté(s).")
        if use_mixed_precision:
            keras.mixed_precision.set_global_policy("mixed_float16")
            print("[HW] Mixed precision activée (float16).")
    else:
        print("[HW] Aucun GPU détecté : entraînement sur CPU (beaucoup plus lent). "
              "Conseil : réduire --final-epochs ou utiliser une machine avec GPU.")


def parse_args():
    p = argparse.ArgumentParser(description="Challenge CIFAR-10 - benchmark, tuning et modèle final")
    p.add_argument("--home-made", action="store_true", help="Lancer uniquement le modèle personnalisé")
    p.add_argument("--phase", choices=["all", "benchmark", "tune", "final"], default="all")
    p.add_argument("--bench-models", nargs="+", default=["mlp", "simple_cnn", "vgg_bn", "resnet9"],
                   choices=list(ModelZoo.REGISTRY))
    p.add_argument("--final-models", nargs="+", default=["vgg_bn", "resnet9"],
                   choices=list(ModelZoo.REGISTRY))
    p.add_argument("--tune-model", default="resnet9", choices=list(ModelZoo.REGISTRY))
    p.add_argument("--subset-size", type=int, default=10000, help="taille du sous-ensemble (benchmark/tuning)")
    p.add_argument("--bench-epochs", type=int, default=15)
    p.add_argument("--final-epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr-grid", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    p.add_argument("--skip-tuning", action="store_true")
    p.add_argument("--no-mixed-precision", action="store_true")
    p.add_argument("--quick", action="store_true", help="exécution de test très rapide")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results_cifar10")
    args = p.parse_args()
    if args.quick:
        args.subset_size, args.bench_epochs, args.final_epochs = 2000, 2, 2
        args.lr_grid = [0.1]
    return args


def main():
    args = parse_args()
    keras.utils.set_random_seed(args.seed)
    setup_hardware(not args.no_mixed_precision)
    Experiment(args).run()


if __name__ == "__main__":
    main()
