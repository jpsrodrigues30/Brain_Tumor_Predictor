from gc import callbacks
import os
import numpy as np
import random
from glob import glob
from typing import Tuple, List
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import traceback

class BrainTumorTrainer:
    def __init__(
            self,
            train_path: str,
            # input_size: Tuple[int, int] = [256, 256],
            input_size: Tuple[int, int] = [224, 224],
            batch_size: int = 32,
            seed: int = 42
    ):
        """
        train_path: Directory with the normalized .npy files [0,1] for training
        test_path: Directory with the normalized .npy files [0,1] for testing
        """
        self.train_path = train_path
        self.input_size = input_size
        self.batch_size = batch_size
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        self.class_names = []
        self.model = None
        self.history = None
        self.X_train = self.X_val = None
        self.y_train = self.y_val = None
    
    def _load_split_data(self, base_directory: str) -> Tuple[np.ndarray, np.ndarray, List[str]] :
        """
        Load all .npy images from base_directory
        """
        class_names = sorted([
            d for d in os.listdir(base_directory)
            if os.path.isdir(os.path.join(base_directory, d))
        ])

        X_list = []
        y_list = []
        total_count = 0

        for idx, cls in enumerate(class_names):
            cls_path = os.path.join(base_directory, cls)

            # Get all all files that ends with .npy from the class directory
            img_files = glob(os.path.join(cls_path, "*.npy"))

            print(f"[DEBUG] classe {cls} -> {len(img_files)} amostras")

            for img in img_files:
                arr = np.load(img) # 
                # Saves the image in the array, alongside with its label
                X_list.append(arr)
                y_list.append(idx)
                total_count+=1
        
            print(f"[DEBUG] total acumulado até agora: {total_count}")

        print("[DEBUG] empilhando tudo em memória com np.stack agora... isso pode demorar")

        X = np.stack(X_list, axis=0).astype("float32")
        y = np.array(y_list, dtype=np.int32)

        print(f"[DEBUG] shape final de X: {X.shape}, y: {y.shape}")
        return X, y, class_names
    
    def load_data(self, val_size: float = 0.2):
        """
        Load training/validation data from self.train_path

        val_size: Fraction of the data that will be used for validation
        """

        X, y, class_names = self._load_split_data(self.train_path)
        self.class_names = class_names

        # Ensures that each class will appear proportionally (stratified)
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size = val_size,
            random_state=self.seed,
            stratify=y
        )

        self.X_train = X_train
        self.y_train = y_train
        self.X_val   = X_val
        self.y_val   = y_val

        print(f"Treino: {self.X_train.shape}, Val: {self.X_val.shape}")
        print(f"Classes: {self.class_names}")

    def _make_dataset(self, X, y, shuffle=True):
        """
        Creates new dataset with
        - Scale converion from [0,1] -> [-1,1] (Scale expected by MobileNetV2)
        - Batch size
        - Prefetch
        """
        ds = tf.data.Dataset.from_tensor_slices((X,y))

        if shuffle:
            ds = ds.shuffle(buffer_size=len(X), seed=self.seed)
        
        def preprocess(x, label):
            x=(x*2.0)-1.0 # Scale conversion
            return x, label
        
        ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds
    
    def build_model(self, lr: float = 1e-04, train_base: bool = False, fine_tune: bool = False):
        """
        Creates new MobileNetV2 model pre-trained
        """
        num_classes = len(self.class_names)

        base_model = MobileNetV2(
            input_shape=(self.input_size[0], self.input_size[1],3),
            include_top=False,
            weights="imagenet"
        )

        # Frozen training base
        base_model.trainable=train_base

        inputs = layers.Input(
            shape=(self.input_size[0], self.input_size[1], 3),
            name="input_image"
        )

        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)

        model = models.Model(inputs, outputs, name="BrainTumorMobileNetV2")

        if fine_tune:
            fine_tune_at_layer = 100 
            for layer_idx, layer in enumerate(model.layers[1].layers):  
                if layer_idx >= fine_tune_at_layer:
                    layer.trainable = True
                else:
                    layer.trainable = False

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        self.model = model
        self.model.summary()

    def train(self, patience: int = 5, min_lr: float = 1e-6, epochs: int = 50):
        """
        Treino com EarlyStopping (para cedo quando overfit começar)
        e ReduceLROnPlateau (diminui LR quando validação estagna).
        """
        print("Starting training...")
        train_ds = self._make_dataset(self.X_train, self.y_train, shuffle=True)
        print("Starting validation...")
        val_ds   = self._make_dataset(self.X_val,   self.y_val,   shuffle=False)

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=max(1, patience // 2),
                min_lr=min_lr
            )
        ]

        print("Testing one batch from train_ds...")
        for batch_x, batch_y in train_ds.take(1):
            print("Batch X shape:", batch_x.shape)
            print("Batch y shape:", batch_y.shape)
        print("Batch OK, calling model.fit...")


        print("About to call model.fit...")
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        print("model.fit finished")

    def _save_results(self):
        """
        Salva o histórico de execução em results/training_results.txt.
        Se ainda não houve treino (self.history == None), salva só info do dataset.
        """    
        os.makedirs("results", exist_ok=True)
        out_path = os.path.join("results", "training_results.txt")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("=== Dataset info ===\n")
            if self.X_train is not None:
                f.write(f"Treino: {self.X_train.shape}\n")
            else:
                f.write("Treino: N/A\n")

            if self.X_val is not None:
                f.write(f"Val:    {self.X_val.shape}\n")
            else:
                f.write("Val:    N/A\n")

            f.write(f"Classes: {self.class_names}\n\n")

            if self.history is None:
                f.write("=== Training history ===\n")
                f.write("Nenhum histórico disponível (modelo não treinado ainda).\n")
                print(f"Histórico salvo em {out_path} (sem history)")
                return

            hist = self.history.history

            loss_list        = hist.get("loss",        [])
            acc_list         = hist.get("accuracy",    hist.get("acc",        []))
            val_loss_list    = hist.get("val_loss",    [])
            val_acc_list     = hist.get("val_accuracy",hist.get("val_acc",    []))
            lr_history       = hist.get("lr",          hist.get("learning_rate", None))

            f.write("=== Training history (por época) ===\n")
            num_epochs_run = len(loss_list)

            for epoch_idx in range(num_epochs_run):
                loss_epoch     = loss_list[epoch_idx]      if epoch_idx < len(loss_list)     else None
                acc_epoch      = acc_list[epoch_idx]       if epoch_idx < len(acc_list)      else None
                val_loss_epoch = val_loss_list[epoch_idx]  if epoch_idx < len(val_loss_list) else None
                val_acc_epoch  = val_acc_list[epoch_idx]   if epoch_idx < len(val_acc_list)  else None

                line = (
                    f"Epoch {epoch_idx+1}: "
                    f"loss={loss_epoch:.4f} "     if loss_epoch is not None else f"Epoch {epoch_idx+1}: loss=N/A "
                )
                line += (
                    f"acc={acc_epoch:.4f} "       if acc_epoch is not None else "acc=N/A "
                )
                line += (
                    f"val_loss={val_loss_epoch:.4f} " if val_loss_epoch is not None else "val_loss=N/A "
                )
                line += (
                    f"val_acc={val_acc_epoch:.4f} "   if val_acc_epoch is not None else "val_acc=N/A "
                )

                if lr_history is not None and epoch_idx < len(lr_history):
                    line += f"lr={lr_history[epoch_idx]:.6f}"

                f.write(line + "\n")

        print(f"Histórico salvo em {out_path}")

    def save(self, out_dir: str = "trained_model"):
        """
        Salva:
        - o modelo treinado (.keras)
        - a lista de classes (classes.txt)
        Isso é importante porque o avaliador/teste precisa saber
        qual índice corresponde a qual classe.
        """
        print("Saving training and validation results...")
        self._save_results()
        os.makedirs(out_dir, exist_ok=True)

        model_path = os.path.join(out_dir, "model.keras")
        self.model.save(model_path)

        classes_path = os.path.join(out_dir, "classes.txt")
        with open(classes_path, "w", encoding="utf-8") as f:
            for cls in self.class_names:
                f.write(cls + "\n")

        print(f"Modelo salvo em: {model_path}")
        print(f"Classes salvas em: {classes_path}")


if __name__ == "__main__":
    trainer = BrainTumorTrainer(
        train_path="dataset/normalized/Training", 
        input_size=(224, 224),
        batch_size=32,
        seed=42
    )

    trainer.load_data(val_size=0.2)

    try:
        print("Building model...")
        trainer.build_model(
            lr=1e-4,
            train_base=False,
            fine_tune=False
        )

        print("Initializing training...")
        trainer.train(
            patience=5,
            min_lr=1e-6,
            epochs=25
        )

        print("Saving...")
        trainer.save(out_dir="trained_model")
    except Exception as e:
        print("An error occurred during training:", e)
        traceback.print_exc()

