import os
import numpy as np
from glob import glob
from typing import Tuple, List
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_curve,
    auc
)
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt


class BrainTumorEvaluator:
    def __init__(
        self,
        test_path: str,
        model_dir: str,
        batch_size: int = 32,
        seed: int = 42
    ):
        """
        test_path: directory with the .npy files reserved for testing
        model_dir: directory with the trained model
        """
        self.test_path = test_path
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.seed = seed

        np.random.seed(seed)
        tf.random.set_seed(seed)

        self.class_names: List[str] = []
        self.model = None

        self.X_test = None
        self.y_test = None

    def _load_split_data(self, base_directory: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Get the testing data
        """
        class_names = sorted([
            d for d in os.listdir(base_directory)
            if os.path.isdir(os.path.join(base_directory, d))
        ])

        X_list = []
        y_list = []

        for idx, cls in enumerate(class_names):
            cls_path = os.path.join(base_directory, cls)
            npy_files = glob(os.path.join(cls_path, "*.npy"))

            for f in npy_files:
                arr = np.load(f) 
                X_list.append(arr)
                y_list.append(idx)

        X = np.stack(X_list, axis=0).astype("float32")
        y = np.array(y_list, dtype=np.int32)

        return X, y, class_names

    def _make_dataset(self, X, y):
        """
        Create a new dataset for evaluation
        """
        ds = tf.data.Dataset.from_tensor_slices((X, y))

        def preprocess(x, label):
            x = (x * 2.0) - 1.0 
            return x, label

        ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def load_test_data(self):
        """
        Loads everything present in the testing directory
        """
        X, y, class_names = self._load_split_data(self.test_path)
        self.X_test = X
        self.y_test = y
       
        print(f"Teste: {self.X_test.shape}")
        print(f"Classes detectadas no diretório de teste: {class_names}")

    def load_model(self):
        """
        Loads the trained model + labels used for classification
        """
        classes_path = os.path.join(self.model_dir, "classes.txt")
        with open(classes_path, "r", encoding="utf-8") as f:
            self.class_names = [line.strip() for line in f.readlines()]

        model_path = os.path.join(self.model_dir, "model.keras")
        self.model = tf.keras.models.load_model(model_path)

        self.model.summary()
        print(f"Classes carregadas do modelo: {self.class_names}")

    def evaluate(self, out_dir: str = "results"):
        """
        Evaluates the testing set and generates performance metrics
        - Accuracy, precision, recall, f1
        - Mcc, kappa
        - Confunsion matrix (jpg)
        - RCO curves per class (jpg)
        - Embeddigs t-SNE (jpg)
        - metrics_test.txt with everything
        """

        os.makedirs(out_dir, exist_ok=True)

        test_ds = self._make_dataset(self.X_test, self.y_test)

        y_prob = self.model.predict(test_ds)
        y_pred = np.argmax(y_prob, axis=1)
        y_true = self.y_test.copy()

        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)

        print("=== MÉTRICAS GERAIS (TESTE) ===")
        print(f"Acurácia:  {acc:.4f}")
        print(f"Precisão:  {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"MCC:       {mcc:.4f}")
        print(f"Kappa:     {kappa:.4f}")

        cm = confusion_matrix(y_true, y_pred)
        print("Matriz de Confusão:")
        print(cm)

        plt.figure(figsize=(6,6))
        plt.imshow(cm, interpolation="nearest")
        plt.title("Matriz de Confusão (Teste)")
        plt.colorbar()
        tick_marks = np.arange(len(self.class_names))
        plt.xticks(tick_marks, self.class_names, rotation=45, ha="right")
        plt.yticks(tick_marks, self.class_names)
        plt.xlabel("Predito")
        plt.ylabel("Real")
        plt.tight_layout()
        cm_path = os.path.join(out_dir, "confusion_matrix_test.jpg")
        plt.savefig(cm_path, dpi=200)
        plt.close()

        y_true_onehot = tf.keras.utils.to_categorical(
            y_true,
            num_classes=len(self.class_names)
        )

        plt.figure(figsize=(6,6))
        for i, cls_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{cls_name} (AUC={roc_auc:.2f})")
        plt.plot([0,1],[0,1],"k--",label="Aleatório")
        plt.xlabel("Falso Positivo")
        plt.ylabel("Verdadeiro Positivo (Recall)")
        plt.legend(loc="lower right")
        plt.title("Curvas ROC por classe (Teste)")
        plt.tight_layout()
        roc_path = os.path.join(out_dir, "roc_curves_test.jpg")
        plt.savefig(roc_path, dpi=200)
        plt.close()

        feature_extractor = tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.layers[-3].output 
        )

        X_test_preproc = (self.X_test * 2.0) - 1.0  
        feats = feature_extractor.predict(
            X_test_preproc,
            batch_size=self.batch_size
        )

        tsne = TSNE(
            n_components=2,
            learning_rate="auto",
            init="random",
            perplexity=30,
            random_state=self.seed
        )
        feats_2d = tsne.fit_transform(feats)

        plt.figure(figsize=(6,6))
        for i, cls_name in enumerate(self.class_names):
            idxs = np.where(y_true == i)[0]
            plt.scatter(
                feats_2d[idxs, 0],
                feats_2d[idxs, 1],
                alpha=0.7,
                label=cls_name,
                s=20
            )
        plt.title("t-SNE dos embeddings (Teste)")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        tsne_path = os.path.join(out_dir, "tsne_embeddings_test.jpg")
        plt.savefig(tsne_path, dpi=200)
        plt.close()

        metrics_path = os.path.join(out_dir, "metrics_test_first_run.txt")
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write("=== MÉTRICAS GERAIS (TESTE) ===\n")
            f.write(f"Acurácia:  {acc:.4f}\n")
            f.write(f"Precisão:  {prec:.4f}\n")
            f.write(f"Recall:    {rec:.4f}\n")
            f.write(f"F1-Score:  {f1:.4f}\n")
            f.write(f"MCC:       {mcc:.4f}\n")
            f.write(f"Kappa:     {kappa:.4f}\n")
            f.write("\nMatriz de confusão:\n")
            f.write(str(cm))
            f.write("\nClasses:\n")
            f.write(str(self.class_names))

        print(f"Relatórios salvos em {out_dir}")

if __name__ == "__main__":
    evaluator = BrainTumorEvaluator(
        test_path="dataset/normalized/Testing",
        model_dir="trained_model",        
        batch_size=32,
        seed=42
    )

    evaluator.load_test_data() 
    evaluator.load_model()    

    evaluator.evaluate(out_dir="results")