# brain.py — Polished EEG Patient Dashboard (NeuralNet only) with accuracy, donut chart & nicer UI
# Run: streamlit run brain.py
# Reference sample report (used for UI examples): features_dataset.csv_EEG_report. :contentReference[oaicite:1]{index=1}

import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, io, traceback
from datetime import datetime
from scipy.signal import butter, filtfilt, welch, iirnotch
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go

# TensorFlow / Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False
    tf = None


NN_MODEL_FILE = "eeg_nn_windowed.h5"
SCALER_FILE = "eeg_scaler_windowed.joblib"
LE_FILE = "eeg_labelenc_windowed.joblib"

FS = 256
WINDOW_SEC = 2.0
STEP_SEC = 1.0
REQUIRED_LABEL_FRACTION = 0.6


def bandpass_filter(sig, low=0.5, high=45, fs=256, order=4):
    nyq = fs / 2
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)

def notch50(sig, fs=256, freq=50.0, Q=30.0):
    try:
        b, a = iirnotch(freq/(fs/2), Q)
        return filtfilt(b, a, sig)
    except Exception:
        return sig

def bandpower(sig, fs, low, high):
    freqs, psd = welch(sig, fs=fs, nperseg=min(512, len(sig)))
    idx = (freqs >= low) & (freqs <= high)
    return np.trapz(psd[idx], freqs[idx])

def hjorth(sig):
    dx = np.diff(sig)
    ddx = np.diff(dx)
    var_x = np.var(sig)
    var_dx = np.var(dx)
    var_ddx = np.var(ddx)
    mobility = np.sqrt(var_dx / (var_x + 1e-12))
    complexity = np.sqrt(var_ddx / (var_dx + 1e-12)) / (mobility + 1e-12)
    return mobility, complexity

def extract_windowed_features_from_signal_array(sig_array, label_array=None,
                                               fs=FS, window_sec=WINDOW_SEC, step_sec=STEP_SEC,
                                               required_label_fraction=REQUIRED_LABEL_FRACTION):
    wlen = int(window_sec * fs)
    step = int(step_sec * fs)
    records = []
    n = len(sig_array)
    for start in range(0, n - wlen + 1, step):
        end = start + wlen
        window = sig_array[start:end]
        if label_array is not None:
            wins = label_array[start:end]
            mode = pd.Series(wins).mode()
            if len(mode) == 0:
                continue
            mode_label = mode[0]
            frac = (wins == mode_label).mean()
            if frac < required_label_fraction:
                continue
        else:
            mode_label = "unknown"

        # preprocess
        try:
            w = bandpass_filter(window, fs=fs)
        except Exception:
            w = window
        try:
            w = notch50(w, fs=fs)
        except Exception:
            pass

        dp = bandpower(w, fs, 0.5, 4)
        tp = bandpower(w, fs, 4, 8)
        ap = bandpower(w, fs, 8, 12)
        bp = bandpower(w, fs, 12, 30)
        gp = bandpower(w, fs, 30, 45)
        tot = dp + tp + ap + bp + gp + 1e-12
        delta = dp / tot; theta = tp / tot; alpha = ap / tot; beta = bp / tot; gamma = gp / tot
        theta_alpha = theta / (alpha + 1e-12)
        beta_alpha = beta / (alpha + 1e-12)
        delta_theta = delta / (theta + 1e-12)

        rms = np.sqrt(np.mean(w**2))
        var = np.var(w)
        sk = float(skew(w))
        kurt = float(kurtosis(w))
        m, c = hjorth(w)

        rec = {
            "delta": delta, "theta": theta, "alpha": alpha, "beta": beta, "gamma": gamma,
            "theta_alpha": theta_alpha, "beta_alpha": beta_alpha, "delta_theta": delta_theta,
            "rms": rms, "var": var, "skew": sk, "kurtosis": kurt, "hjorth_mobility": m, "hjorth_complexity": c,
            "label": mode_label,
            "start_sample": int(start), "end_sample": int(end)
        }
        records.append(rec)
    return pd.DataFrame(records)


def create_pdf_report(filename, report):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, "EEG Patient Report", ln=True, align="C")
    pdf.ln(4)

    pdf.set_font("Arial", size=10)
    pdf.cell(0, 6, f"File: {filename}", ln=True)
    pdf.cell(0, 6, f"Windows analyzed: {report.get('n_windows', 0)}", ln=True)
    pdf.ln(4)

    pdf.cell(0, 6, "Mean probabilities (per class):", ln=True)
    for k, v in report.get('mean_probs', {}).items():
        pdf.cell(0, 6, f"  - {k}: {v*100:.1f}%", ln=True)

    pdf.ln(4)
    pdf.cell(0, 6, "Majority window percentage:", ln=True)
    for k, v in report.get('majority_pct', {}).items():
        pdf.cell(0, 6, f"  - {k}: {v:.1f}%", ln=True)

    pdf.ln(6)
    pdf.cell(0, 6, "Automated notes (research tool, non-diagnostic):", ln=True)
    if report.get('mean_probs', {}).get("anxiety", 0) > 0.5:
        pdf.multi_cell(0, 6, "- Elevated anxiety-related EEG markers. Consider clinical screening.")
    if report.get('mean_probs', {}).get("depression", 0) > 0.5:
        pdf.multi_cell(0, 6, "- Elevated depression-related EEG markers. Consider mood assessment.")
    if report.get('mean_probs', {}).get("neutral", 0) > 0.6:
        pdf.multi_cell(0, 6, "- EEG appears neutral for this session.")

    pdf.ln(4)
    pdf.cell(0, 6, "Lifestyle Recommendations (Non-diagnostic):", ln=True)
    mean_probs = report.get('mean_probs', {})
    if mean_probs.get("anxiety", 0) > 0.5:
        pdf.multi_cell(0, 6,
            "- Your EEG shows anxiety-related activity.\n"
            "  Recommended: Practice deep breathing (4-7-8), short meditation sessions, reduce caffeine,\n"
            "  20–30 minutes of walking daily, and journaling before sleep.\n")
    if mean_probs.get("depression", 0) > 0.5:
        pdf.multi_cell(0, 6,
            "- Depression-like EEG patterns detected.\n"
            "  Recommended: Morning sunlight exposure (10–15 min), regular sleep schedule, light exercise (yoga/walk),\n"
            "  talk with a friend or counselor, and small activity goals each day to build routine.\n")
    if mean_probs.get("neutral", 0) > 0.6:
        pdf.multi_cell(0, 6,
            "- EEG appears neutral and stable.\n"
            "  Maintain healthy habits: consistent sleep, hydration, balanced meals, and regular physical activity.\n")

    pdf.ln(6)
    pdf.cell(0, 6, f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    return pdf.output(dest='S').encode('latin1')


@st.cache_resource(show_spinner=False)
def load_artifacts():
    nn_model = None; scaler = None; le = None
    try:
        if os.path.exists(NN_MODEL_FILE) and TF_AVAILABLE:
            nn_model = load_model(NN_MODEL_FILE)
    except Exception:
        nn_model = None
    try:
        if os.path.exists(SCALER_FILE):
            scaler = joblib.load(SCALER_FILE)
    except Exception:
        scaler = None
    try:
        if os.path.exists(LE_FILE):
            le = joblib.load(LE_FILE)
    except Exception:
        le = None
    return nn_model, scaler, le

@st.cache_data(show_spinner=False)
def train_neuralnet_on_windowed(df_windowed, epochs=40, batch_size=32, verbose=0):
    if not TF_AVAILABLE:
        st.error("TensorFlow not installed.")
        return None, None, None, None

    X = df_windowed.drop(columns=["label","start_sample","end_sample"], errors='ignore').values
    y = df_windowed["label"].astype(str).values
    le = LabelEncoder(); y_enc = le.fit_transform(y)
    n_classes = len(np.unique(y_enc))
    y_cat = tf.keras.utils.to_categorical(y_enc, num_classes=n_classes)

    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y_cat, test_size=0.2, stratify=y_enc, random_state=42)

    input_dim = Xtr.shape[1]
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    history = model.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=verbose)

    loss, acc = model.evaluate(Xte, yte, verbose=0)
    model.save(NN_MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(le, LE_FILE)
    return model, scaler, le, history

st.set_page_config(page_title="EEG Dashboard — NeuralNet", layout="wide")
st.title("EEG Patient Dashboard — Neural Network")
st.caption("NeuralNet only. Research tool — not a medical device.")

# Load artifacts
nn_model, scaler, le = load_artifacts()
if nn_model is None:
    st.info("No saved NN model found. Upload labeled data to train in-app or train offline.")

# Sidebar
st.sidebar.header("Upload & Controls")
uploaded_files = st.sidebar.file_uploader("Upload CSV files (raw per-sample with `Channel1` or windowed features)", accept_multiple_files=True, type=["csv"])
train_if_missing = st.sidebar.checkbox("Train NeuralNet if artifacts missing", value=False)
show_raw_preview = st.sidebar.checkbox("Show windowed features preview", value=True)
nn_epochs = st.sidebar.number_input("NN epochs", min_value=5, max_value=200, value=40, step=5)
nn_batch = st.sidebar.number_input("NN batch size", min_value=8, max_value=256, value=32, step=8)
nn_verbose = st.sidebar.checkbox("Show NN training logs", value=False)

# Optional: sample report preview (from uploaded sample file)
with st.expander("Example report (uploaded)"):
    st.write("Reference report used for layout and wording.")
    st.markdown("- Uploaded report: features_dataset.csv_EEG_report (sample).")
    # show parsed summary if available (from file search)
    st.markdown("See your uploaded sample report for expected output. :contentReference[oaicite:2]{index=2}")

# Main processing
if uploaded_files:
    st.sidebar.success(f"{len(uploaded_files)} file(s) ready")
    if st.sidebar.button("Process & Generate Reports"):
        for up in uploaded_files:
            with st.spinner(f"Processing {up.name}"):
                try:
                    df_raw = pd.read_csv(up)
                except Exception as e:
                    st.error(f"Failed to read {up.name}: {e}")
                    continue

                # windowing detection
                if set(["delta","theta","alpha","beta"]).issubset(df_raw.columns):
                    df_win = df_raw.copy()
                    if "label" not in df_win.columns:
                        df_win["label"] = "unknown"
                else:
                    if "Channel1" not in df_raw.columns:
                        st.error(f"{up.name}: missing Channel1 and no windowed features.")
                        continue
                    labels = df_raw["label"].values if "label" in df_raw.columns else None
                    sig = df_raw["Channel1"].astype(float).values
                    if labels is not None:
                        labels = labels.astype(str)
                    df_win = extract_windowed_features_from_signal_array(sig, label_array=labels)
                    if df_win.shape[0] == 0:
                        st.warning("No confident windows — try changing window/label params.")
                        continue

                # preview
                if show_raw_preview:
                    st.subheader(f"Windowed features — {up.name}")
                    st.dataframe(df_win.head(8))

                # ensure model exists or train if requested
                current_nn = nn_model; current_scaler = scaler; current_le = le
                trained_history = None
                if current_nn is None and train_if_missing:
                    if not TF_AVAILABLE:
                        st.error("TensorFlow not installed — cannot train.")
                        continue
                    if df_win.shape[0] > 50 and df_win['label'].nunique() >= 2:
                        current_nn, current_scaler, current_le, trained_history = train_neuralnet_on_windowed(df_win, epochs=int(nn_epochs), batch_size=int(nn_batch), verbose=1 if nn_verbose else 0)
                        nn_model, scaler, le = current_nn, current_scaler, current_le
                    else:
                        st.warning("Need >50 windows and >=2 classes to auto-train.")
                        continue

                if current_nn is None:
                    st.error("No NN model available. Upload labeled data and enable training or train offline.")
                    continue
                if current_scaler is None or current_le is None:
                    st.error("Scaler or labelencoder missing. Retrain the model to create artifacts.")
                    continue

                # Prepare features & predictions
                feat_cols = [c for c in df_win.columns if c not in ("label","start_sample","end_sample")]
                X = df_win[feat_cols].values
                Xs = current_scaler.transform(X)
                class_names = list(current_le.classes_)

                # Evaluation section (if labeled data present)
                labeled_mask = df_win['label'].astype(str) != "unknown"
                n_label_classes = df_win['label'].nunique() if labeled_mask.any() else 0

                eval_col1, eval_col2 = st.columns([1,2])
                with eval_col1:
                    st.subheader("Model Status")
                    st.markdown(f"- Model file: `{NN_MODEL_FILE}`")
                    st.markdown(f"- Classes: **{', '.join(class_names)}**")
                with eval_col2:
                    if labeled_mask.any() and n_label_classes >= 2:
                        try:
                            y_all = df_win['label'].astype(str).values
                            # transform using saved encoder
                            y_enc_all = current_le.transform(y_all)
                            Xtr_eval, Xte_eval, ytr_eval, yte_eval = train_test_split(Xs, y_enc_all, test_size=0.2, stratify=y_enc_all, random_state=42)
                            yte_cat = tf.keras.utils.to_categorical(yte_eval, num_classes=len(class_names))
                            loss_eval, acc_eval = current_nn.evaluate(Xte_eval, yte_cat, verbose=0)
                            st.metric(label="Test Accuracy", value=f"{acc_eval*100:.2f}%")
                            # classification report
                            y_pred_proba_eval = current_nn.predict(Xte_eval)
                            y_pred_idx_eval = np.argmax(y_pred_proba_eval, axis=1)
                            y_test_labels = current_le.inverse_transform(yte_eval)
                            y_pred_labels = current_le.inverse_transform(y_pred_idx_eval)
                            cr = classification_report(y_test_labels, y_pred_labels, zero_division=0, output_dict=True)
                            cr_df = pd.DataFrame(cr).transpose()
                            st.markdown("**Classification report (test split)**")
                            st.dataframe(cr_df.style.format({"precision": "{:.2f}", "recall":"{:.2f}", "f1-score":"{:.2f}"}))
                            # confusion matrix
                            cm = confusion_matrix(y_test_labels, y_pred_labels, labels=class_names)
                            cm_fig = go.Figure(data=go.Heatmap(z=cm, x=class_names, y=class_names, colorscale='Reds'))
                            cm_fig.update_layout(title="Confusion Matrix (test split)", xaxis_title="Predicted", yaxis_title="True")
                            st.plotly_chart(cm_fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Evaluation failed: {e}")
                    else:
                        st.info("No labeled data for evaluation. Provide labels or upload labeled windowed features.")

                # show training history plots if available immediately after training
                if trained_history is not None:
                    hist = trained_history.history
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Scatter(y=hist['accuracy'], name='train acc'))
                    fig_hist.add_trace(go.Scatter(y=hist['val_accuracy'], name='val acc'))
                    fig_hist.update_layout(title="Training / Validation Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy")
                    st.plotly_chart(fig_hist, use_container_width=True)
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(y=hist['loss'], name='train loss'))
                    fig_loss.add_trace(go.Scatter(y=hist['val_loss'], name='val loss'))
                    fig_loss.update_layout(title="Training / Validation Loss", xaxis_title="Epoch", yaxis_title="Loss")
                    st.plotly_chart(fig_loss, use_container_width=True)

                # per-window predictions
                preds = current_nn.predict(Xs)
                probs_df = pd.DataFrame(preds, columns=class_names)
                pred_df = probs_df.copy()
                pred_df["pred_label"] = pred_df.idxmax(axis=1)
                pred_df["start_sample"] = df_win.get("start_sample", np.nan)
                pred_df["end_sample"] = df_win.get("end_sample", np.nan)

                mean_probs = pred_df[class_names].mean().to_dict()
                majority_pct = pred_df["pred_label"].value_counts(normalize=True).to_dict()
                majority_pct_percent = {k: v*100 for k,v in majority_pct.items()}

                report = {
                    "file": up.name,
                    "n_windows": len(pred_df),
                    "mean_probs": mean_probs,
                    "majority_pct": {k: float(v*100) for k,v in majority_pct.items()},
                    "pred_df": pred_df,
                    "windowed_df": df_win
                }

                # top row: accuracy + donut chart
                a1, a2 = st.columns([1,1])
                with a1:
                    if labeled_mask.any() and n_label_classes >= 2:
                        st.metric("Test Accuracy", f"{acc_eval*100:.2f}%")
                    else:
                        st.info("Test Accuracy: N/A (no labels)")
                with a2:
                    # Pie / donut chart of majority distribution
                    if len(majority_pct_percent) > 0:
                        pie_fig = px.pie(
                            names=list(majority_pct_percent.keys()),
                            values=list(majority_pct_percent.values()),
                            title="Majority label distribution (windows)",
                            hole=0.45
                        )
                        pie_fig.update_traces(textinfo='percent+label')
                        st.plotly_chart(pie_fig, use_container_width=True)
                    else:
                        st.info("Not enough predictions to show distribution.")

                # probability timeline
                st.subheader("Probability timeline (first windows)")
                limit = min(len(pred_df), 300)
                t = np.arange(limit) * STEP_SEC
                timeline_fig = go.Figure()
                for cls in class_names:
                    timeline_fig.add_trace(go.Scatter(x=t, y=pred_df[cls].values[:limit]*100, mode='lines', name=cls))
                timeline_fig.update_layout(yaxis_title="Probability (%)", xaxis_title="Seconds (approx)", yaxis_range=[0,100])
                st.plotly_chart(timeline_fig, use_container_width=True)

                # download per-window CSV
                csv_buf = io.StringIO()
                pred_df.to_csv(csv_buf, index=False)
                st.download_button("Download per-window predictions (CSV)", data=csv_buf.getvalue().encode(), file_name=f"{up.name}_predictions.csv", mime="text/csv")

                # summary + lifestyle suggestions
                st.subheader("Automated summary (for clinician review)")
                avg_text = ", ".join([f"{k}: {v*100:.1f}%" for k,v in mean_probs.items()])
                st.markdown(f"**Windows analyzed:** {len(pred_df)}")
                st.markdown(f"**Mean probabilities:** {avg_text}")
                st.markdown("**Majority of windows predicted:** " + ", ".join([f"{k}: {v:.1f}%" for k,v in report['majority_pct'].items()]))

                st.markdown("### 🌿 Lifestyle Wellness Suggestions (Not diagnostic)")
                aids = []
                if mean_probs.get("anxiety", 0) > 0.5:
                    aids.append("- High anxiety markers detected. Try deep breathing (4-7-8), short meditation, reducing caffeine, 20–30 min walks, and journaling before bed.")
                if mean_probs.get("depression", 0) > 0.5:
                    aids.append("- Depression-like EEG activity observed. Try morning sunlight exposure (10–15 min), regular sleep, light exercise, small daily goals, and talking to someone you trust.")
                if mean_probs.get("neutral", 0) > 0.6:
                    aids.append("- EEG appears neutral. Maintain healthy habits: hydration, regular sleep, exercise, and outdoor time.")
                if len(aids) == 0:
                    aids.append("- No strong emotional stress markers detected — interpret with clinical context.")
                for a in aids:
                    st.markdown(a)

                # PDF
                try:
                    pdf_bytes = create_pdf_report(up.name, report)
                    st.download_button(label=f"Download PDF report for {up.name}", data=pdf_bytes, file_name=f"{up.name}_EEG_report.pdf", mime="application/pdf")
                except Exception as e:
                    st.error(f"Failed to create PDF: {e}")
                    st.text(traceback.format_exc())

        st.success("Done — scroll up to view the results and download reports.")
else:
    st.info("Upload one or more EEG CSV files to begin (raw per-sample with Channel1 or precomputed windowed features).")

st.markdown("---")
st.caption("This tool is experimental and intended for research/clinical-assistive use only. It is not a medical device and cannot replace clinician judgment.")
