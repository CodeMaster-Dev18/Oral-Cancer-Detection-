import tkinter
from tkinter import *
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import lightgbm as lgb
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
import threading
import matplotlib.pyplot as plt


# -----------------------
# File paths / constants
# -----------------------
LIGHTGBM_MODEL_FILE = "lightgbm_model.joblib"
DEEP_SCALER_FILE = "deep_scaler.joblib"
DEEP_PCA_FILE = "deep_pca.joblib"
DEEP_SVM_FILE = "deep_svm.joblib"
DEEP_FEATURES_CACHE = "deep_features.npz"
HOG_FEATURES_CACHE = "hog_features.npz"

# -----------------------
# Globals for dataset/splits
# -----------------------
dataset_path = None

# HOG pipeline arrays and splits
hog_X_train = hog_X_test = None
hog_y_train = hog_y_test = None

# Deep pipeline arrays and splits (features extracted by MobileNet)
deep_X_train = deep_X_test = None
deep_y_train = deep_y_test = None

# Models
lightgbm_clf = None       # LGBMClassifier (sklearn wrapper) saved with joblib
deep_scaler = None
deep_pca = None
deep_svm = None

# MobileNet feature extractor
mobilenet_model = None

# -----------------------
# Utilities
# -----------------------
def get_mobilenet_feature_extractor():
    global mobilenet_model
    if mobilenet_model is None:
        # pooling='avg' yields a 1280-dim vector
        mobilenet_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224,224,3))
    return mobilenet_model

def extract_hog_features_from_cv_image(cv_img):
    resized_image = cv2.resize(cv_img, (128, 128))
    features, _ = hog(resized_image, pixels_per_cell=(16, 16),
                      cells_per_block=(2, 2), visualize=True, multichannel=True)
    return features

def extract_deep_feature_from_cv_image(cv_img):
    # BGR -> RGB
    img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    x = np.expand_dims(img_resized.astype('float32'), axis=0)
    x = mobilenet_preprocess(x)
    model = get_mobilenet_feature_extractor()
    feats = model.predict(x, verbose=0)
    return feats.ravel()  # shape (1280,)

# -----------------------
# GUI callbacks (step order)
# -----------------------
def upload_dataset():
    global dataset_path
    text.delete('1.0',END)
    dataset_path = filedialog.askdirectory(initialdir='.')
    if dataset_path:
        text.insert(tk.END, f"Dataset Uploaded: {dataset_path}\n")
        # show counts
        class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        for cls in class_dirs:
            count = len([f for f in os.listdir(os.path.join(dataset_path, cls)) if os.path.isfile(os.path.join(dataset_path, cls, f))])
            text.insert(tk.END, f"{cls}: {count}\n")
    else:
        text.insert(tk.END, "No dataset selected.\n")

def preprocess_dataset():
    """
    Builds:
      - HOG features and train/test split saved to hog_X_train/test, hog_y_train/test
      - Deep features (MobileNet) cached to DEEP_FEATURES_CACHE and split saved to deep_X_train/test, deep_y_train/test
    """
    def task():
        global hog_X_train, hog_X_test, hog_y_train, hog_y_test
        global deep_X_train, deep_X_test, deep_y_train, deep_y_test
        text.delete('1.0',END)

        if dataset_path is None:
            messagebox.showerror("Error", "Please upload the dataset first.")
            return

        text.insert(tk.END, "Preprocessing started...\n")
        text.see(tk.END)

        # HOG extraction (optionally cache)
        if os.path.exists(HOG_FEATURES_CACHE):
            try:
                data = np.load(HOG_FEATURES_CACHE, allow_pickle=True)
                hog_X = data['X']
                hog_y = data['y']
                text.insert(tk.END, f"Loaded HOG features from cache ({HOG_FEATURES_CACHE}).\n")
            except Exception as e:
                text.insert(tk.END, f"Failed to load HOG cache, re-extracting. Err: {e}\n")
                hog_X, hog_y = extract_and_cache_hog_features()
        else:
            hog_X, hog_y = extract_and_cache_hog_features()

        # split HOG
        hog_X_train, hog_X_test, hog_y_train, hog_y_test = train_test_split(hog_X, hog_y, test_size=0.2, random_state=42, stratify=hog_y)
        text.insert(tk.END, f"HOG split: train={len(hog_X_train)} test={len(hog_X_test)}\n")

        # Deep features extraction (cache)
        if os.path.exists(DEEP_FEATURES_CACHE):
            try:
                data = np.load(DEEP_FEATURES_CACHE, allow_pickle=True)
                deep_X = data['X']
                deep_y = data['y']
                text.insert(tk.END, f"Loaded deep features from cache ({DEEP_FEATURES_CACHE}).\n")
            except Exception as e:
                text.insert(tk.END, f"Failed to load deep cache, re-extracting. Err: {e}\n")
                deep_X, deep_y = extract_and_cache_deep_features()
        else:
            deep_X, deep_y = extract_and_cache_deep_features()

        # split deep
        deep_X_train, deep_X_test, deep_y_train, deep_y_test = train_test_split(deep_X, deep_y, test_size=0.2, random_state=42, stratify=deep_y)
        text.insert(tk.END, f"Deep split: train={len(deep_X_train)} test={len(deep_X_test)}\n")

        text.insert(tk.END, "Preprocessing completed.\n")
        text.see(tk.END)

    run_in_thread(task)

def extract_and_cache_hog_features():
    """
    Walk dataset, extract HOG for each image, return X (n_samples, feat_dim), y (n_samples,)
    Also save to HOG_FEATURES_CACHE
    """
    items = []
    labels = []
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue
        for fname in os.listdir(class_path):
            fpath = os.path.join(class_path, fname)
            if not os.path.isfile(fpath):
                continue
            img = cv2.imread(fpath)
            if img is None:
                continue
            feats = extract_hog_features_from_cv_image(img)
            items.append(feats)
            labels.append(0 if class_name.upper() == "NON CANCER" else 1)
    X = np.array(items)
    y = np.array(labels)
    np.savez_compressed(HOG_FEATURES_CACHE, X=X, y=y)
    text.insert(tk.END, f"Saved HOG features to cache: {HOG_FEATURES_CACHE}\n")
    return X, y

def extract_and_cache_deep_features():
    """
    Walk dataset, extract MobileNet features for each image, save to DEEP_FEATURES_CACHE,
    return X (n_samples, feat_dim), y (n_samples,)
    """
    items = []
    labels = []
    model = get_mobilenet_feature_extractor()
    image_paths = []
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue
        for fname in os.listdir(class_path):
            fpath = os.path.join(class_path, fname)
            if not os.path.isfile(fpath):
                continue
            image_paths.append((fpath, class_name))
    total = len(image_paths)
    text.insert(tk.END, f"Extracting deep features for {total} images...\n")
    text.see(tk.END)
    for idx, (p, cls) in enumerate(image_paths):
        img = cv2.imread(p)
        if img is None:
            feats = np.zeros((model.output_shape[1],), dtype=np.float32)
        else:
            feats = extract_deep_feature_from_cv_image(img)
        items.append(feats)
        labels.append(0 if cls.upper() == "NON CANCER" else 1)
        if (idx+1) % 50 == 0:
            text.insert(tk.END, f"Extracted {idx+1}/{total}\n")
            text.see(tk.END)
    X = np.vstack(items)
    y = np.array(labels)
    np.savez_compressed(DEEP_FEATURES_CACHE, X=X, y=y)
    text.insert(tk.END, f"Saved deep features to cache: {DEEP_FEATURES_CACHE}\n")
    return X, y

# -----------------------
# Train LightGBM (on HOG features)
# -----------------------
def train_lightgbm():
    def task():
        global lightgbm_clf, hog_X_train, hog_X_test, hog_y_train, hog_y_test
        text.delete('1.0',END)
        if hog_X_train is None or hog_y_train is None:
            messagebox.showerror("Error", "Please run Preprocess Dataset first.")
            return
        text.insert(tk.END, "Training LightGBM on HOG features...\n")
        text.see(tk.END)

        # Use LGBMClassifier (sklearn wrapper) for easy joblib saving
        lgbm = lgb.LGBMClassifier(objective='binary', learning_rate=0.05, n_estimators=500,
                                  num_leaves=31, class_weight='balanced')
        lgbm.fit(hog_X_train, hog_y_train)

        # Save
        joblib.dump(lgbm, LIGHTGBM_MODEL_FILE)
        lightgbm_clf = lgbm

        # Evaluate on HOG test set
        preds = lgbm.predict(hog_X_test)
        acc = accuracy_score(hog_y_test, preds)
        prec = precision_score(hog_y_test, preds, zero_division=0)
        rec = recall_score(hog_y_test, preds, zero_division=0)
        f1 = f1_score(hog_y_test, preds, zero_division=0)
        cm = confusion_matrix(hog_y_test, preds)

        text.insert(tk.END, "LightGBM metrics:\n")
        text.insert(tk.END, f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\n")
        text.insert(tk.END, f"Confusion Matrix:\n{cm}\n")
        text.see(tk.END)

    run_in_thread(task)

# -----------------------
# Train Deep Model (MobileNetV2 -> Scaler -> PCA -> SVM)
# -----------------------
def train_deep_model():
    def task():
        global deep_scaler, deep_pca, deep_svm, deep_X_train, deep_X_test, deep_y_train, deep_y_test
        text.delete('1.0',END)
        if deep_X_train is None:
            messagebox.showerror("Error", "Please run Preprocess Dataset first.")
            return
        text.insert(tk.END, "Training Deep Model (MobileNetV2 features + SVM)...\n")
        text.see(tk.END)

        # Preprocessing
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(deep_X_train)
        X_test_scaled = scaler.transform(deep_X_test)

        # PCA reduction
        pca = PCA(n_components=0.95, svd_solver='full')
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        text.insert(tk.END, f"PCA reduced features: {deep_X_train.shape[1]} -> {X_train_pca.shape[1]}\n")
        text.see(tk.END)

        # SVM training
        svm = SVC(kernel='rbf', probability=True, class_weight='balanced', C=1.0, gamma='scale')
        svm.fit(X_train_pca, deep_y_train)

        # Save pipeline
        joblib.dump(scaler, DEEP_SCALER_FILE)
        joblib.dump(pca, DEEP_PCA_FILE)
        joblib.dump(svm, DEEP_SVM_FILE)
        deep_scaler = scaler
        deep_pca = pca
        deep_svm = svm

        # Evaluate
        preds = svm.predict(X_test_pca)
        acc = accuracy_score(deep_y_test, preds)
        prec = precision_score(deep_y_test, preds, zero_division=0)
        rec = recall_score(deep_y_test, preds, zero_division=0)
        f1 = f1_score(deep_y_test, preds, zero_division=0)
        cm = confusion_matrix(deep_y_test, preds)

        text.insert(tk.END, "Deep Model metrics:\n")
        text.insert(tk.END, f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\n")
        text.insert(tk.END, f"Confusion Matrix:\n{cm}\n")
        text.see(tk.END)

    run_in_thread(task)

# -----------------------
# Comparison Graph
# -----------------------
def comparison_graph():
    """
    Loads metrics from saved models or from current variables and shows a bar graph
    for Accuracy / Precision / Recall / F1 for both models.
    """
    def task():
        text.delete('1.0',END)
        # Gather metrics for LightGBM
        hog_metrics = None
        deep_metrics = None

        # LightGBM
        try:
            if lightgbm_clf is None and os.path.exists(LIGHTGBM_MODEL_FILE):
                temp = joblib.load(LIGHTGBM_MODEL_FILE)
            else:
                temp = lightgbm_clf

            if temp is not None and hog_X_test is not None:
                preds = temp.predict(hog_X_test)
                hog_metrics = compute_metrics(hog_y_test, preds)
        except Exception as e:
            text.insert(tk.END, f"Could not compute LightGBM metrics for graph: {e}\n")

        # Deep
        try:
            if deep_svm is None and os.path.exists(DEEP_SVM_FILE):
                scaler = joblib.load(DEEP_SCALER_FILE)
                pca = joblib.load(DEEP_PCA_FILE)
                svm = joblib.load(DEEP_SVM_FILE)
                # compute on deep_X_test if available
                if deep_X_test is not None and deep_y_test is not None:
                    X_ts_scaled = scaler.transform(deep_X_test)
                    X_ts_pca = pca.transform(X_ts_scaled)
                    preds = svm.predict(X_ts_pca)
                    deep_metrics = compute_metrics(deep_y_test, preds)
            elif deep_svm is not None and deep_X_test is not None:
                X_ts_scaled = deep_scaler.transform(deep_X_test)
                X_ts_pca = deep_pca.transform(X_ts_scaled)
                preds = deep_svm.predict(X_ts_pca)
                deep_metrics = compute_metrics(deep_y_test, preds)
        except Exception as e:
            text.insert(tk.END, f"Could not compute Deep model metrics for graph: {e}\n")

        if hog_metrics is None and deep_metrics is None:
            messagebox.showinfo("Info", "No metrics available to plot. Train models first.")
            return

        # Prepare data for plot
        labels = ["Accuracy", "Precision", "Recall", "F1"]
        hog_vals = [hog_metrics.get(k, 0) for k in labels] if hog_metrics else [0,0,0,0]
        deep_vals = [deep_metrics.get(k, 0) for k in labels] if deep_metrics else [0,0,0,0]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots()
        ax.bar(x - width/2, hog_vals, width, label='LightGBM (HOG)')
        ax.bar(x + width/2, deep_vals, width, label='Deep (MobileNetV2+SVM)')
        ax.set_ylabel('Score')
        ax.set_title('Model comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0,1.0)
        ax.legend()
        for i, v in enumerate(hog_vals):
            ax.text(i - width/2, v + 0.01, f"{v:.2f}", ha='center')
        for i, v in enumerate(deep_vals):
            ax.text(i + width/2, v + 0.01, f"{v:.2f}", ha='center')

        plt.show()

    run_in_thread(task)

def compute_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0)
    }

def predict():
    def task():
        global lightgbm_clf
        text.delete('1.0',END)
        if lightgbm_clf is None and os.path.exists(LIGHTGBM_MODEL_FILE):
            try:
                lightgbm_clf = joblib.load(LIGHTGBM_MODEL_FILE)
                text.insert(tk.END, "Loaded LightGBM model for prediction.\n")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load LightGBM model: {e}")
                return
        if lightgbm_clf is None:
            messagebox.showerror("Error", "LightGBM model not trained/saved. Train LightGBM first.")
            return

        test_image_path = filedialog.askopenfilename(title="Select Test Image", filetypes=[("Image Files", "*.jpeg *.jpg *.png")])
        if not test_image_path:
            return
        img = cv2.imread(test_image_path)
        if img is None:
            messagebox.showerror("Error", "Unable to open image.")
            return
        feats = extract_hog_features_from_cv_image(img).reshape(1, -1)
        pred = lightgbm_clf.predict(feats)
        label = "CANCER" if int(pred[0]) == 1 else "NON CANCER"
        text.insert(tk.END, f"Prediction: {label}\n")
        display_image_with_prediction(img, label)

    run_in_thread(task)

# -----------------------
# Image display utility
# -----------------------
def display_image_with_prediction(image, prediction):
    scale_factor = 1.2
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    resized_image = cv2.resize(image, (new_width, new_height))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(resized_image, f"Prediction: {prediction}", (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Result", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -----------------------
# Helper to run in thread
# -----------------------
def run_in_thread(target_fn):
    t = threading.Thread(target=target_fn, daemon=True)
    t.start()

# -----------------------
# GUI build (buttons in requested order)
# -----------------------


main=tkinter.Tk()
main.title("Oral Cancer")
main.geometry("1200x1000")



font2=('Comic Sans MS',20,'bold')
title=Label(main,text="ORAL CANCER DETECTION FRAMEWORK")
title.config(bg="gray",fg="White")
title.config(font=font2)
title.config(height=3,width=120)
title.place(x=0,y=5, relwidth=1)

font1=('Comic Sans MS',13,'bold')
text=Text(main,height=18,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.config(font=font1)
text.place(x=0,y=125)
text.config(bg="black",fg="White")



button1=Button(main,text="Upload Dataset", command=upload_dataset)
button1.config(font=font1)
button1.place(x=10,y=600)

button2=Button(main,text="Preprocess Dataset", command=preprocess_dataset)
button2.config(font=font1)
button2.place(x=160,y=600)

button3=Button(main,text="Train LightGBM", command=train_lightgbm)
button3.config(font=font1)
button3.place(x=360,y=600)

button4=Button(main,text="Train Deep Model (MobileNetV2 + SVM)",command=train_deep_model)
button4.config(font=font1)
button4.place(x=550,y=600)

button5=Button(main,text="Comparison Graph", command=comparison_graph)
button5.config(font=font1)
button5.place(x=950,y=600)

button6=Button(main,text="Predict", command=predict)
button6.config(font=font1)
button6.place(x=1150,y=600)


main.config(bg="pink")
main.mainloop()
