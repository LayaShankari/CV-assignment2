"""
app.py
------
Interactive Streamlit Application:
"Fashion-MNIST: Nearest Neighbor vs Histogram Classifier for Low-Resolution Image Classification"
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from src.data import load_fashion_mnist, LABELS
from src.nearest_neighbor import (
    train_nearest_neighbor,
    predict_one_nearest_neighbor,
    find_closest_training_example,
)
from src.histogram_classifier import (
    train_histogram_classifier,
    predict_one_histogram,
    get_histogram_distances,
)
from src.evaluation import evaluate_nearest_neighbor, evaluate_histogram
from src.preprocessing import preprocess_uploaded_image

st.set_page_config(
    page_title="Fashion-MNIST Classifier Comparison",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("👕 Fashion-MNIST: Nearest Neighbor vs. Histogram Classifier")
st.markdown(
    "A comparative study of two traditional computer vision techniques for low-resolution "
    "($14 \\times 14$ pixel) clothing classification on the **Fashion-MNIST** benchmark dataset."
)

with st.sidebar:
    st.header("⚙️ Project Settings")
    train_count = st.selectbox("Training Samples", [5000, 10000, 15000], index=1)
    test_count = st.selectbox("Evaluation Samples", [500, 1000, 2000], index=1)
    st.caption("Larger sample sizes yield higher evaluation accuracy but take slightly longer to compute.")


@st.cache_resource(show_spinner="Training Nearest Neighbor and Histogram classifiers...")
def build_models(train_count):
    x_train, y_train, x_test, y_test = load_fashion_mnist(train_count, 2000)
    nn_model = train_nearest_neighbor(x_train, y_train)
    hist_templates = train_histogram_classifier(x_train, y_train)
    return nn_model, hist_templates, x_train, y_train


x_train, y_train, x_test, y_test = load_fashion_mnist(train_count, test_count)
nn_model, hist_templates, full_x_train, full_y_train = build_models(train_count)

tab1, tab2, tab3 = st.tabs(["📖 Overview & Dataset", "📊 Compare Performance", "🎯 Test Clothing Prediction"])

# ==============================================================================
# TAB 1: OVERVIEW & DATASET
# ==============================================================================
with tab1:
    st.subheader("Dataset Samples (Downsampled to 14 × 14)")
    st.markdown(
        "The Fashion-MNIST dataset contains $28 \\times 28$ grayscale images of clothing items. "
        "For this low-resolution benchmark, images are subsampled to **$14 \\times 14$ pixels** (196 total pixels)."
    )

    sample_indices = np.arange(10)
    cols = st.columns(10)
    for i, col in zip(sample_indices, cols):
        with col:
            st.image(x_train[i], caption=f"{LABELS[int(y_train[i])]}", clamp=True, width=80)

    st.info(
        f"✅ Loaded dataset subset: **{len(x_train):,}** training images and **{len(x_test):,}** testing images. "
        f"Input shape per image: `(14, 14)`."
    )

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("### 1-Nearest Neighbor (Pixel-Based)")
        st.markdown(
            """
            - **How it works**: Computes the Euclidean Distance between test image vector $\\mathbf{x}$ and all training vectors $\\mathbf{y}$:
              $$d(\\mathbf{x}, \\mathbf{y}) = \\sqrt{\\sum_{i=1}^{196} (x_i - y_i)^2}$$
            - **Strength**: Preserves **spatial structure** (where sleeves, collars, soles, or zippers appear).
            - **Typical Accuracy**: ~**75% – 82%** on test data.
            """
        )
    with col_info2:
        st.markdown("### Histogram Classifier (Intensity-Based)")
        st.markdown(
            """
            - **How it works**: Counts grayscale frequencies into 16 bins and computes Chi-Square distance to average class templates:
              $$\\chi^2(P, Q) = \\frac{1}{2} \\sum_{i=1}^{16} \\frac{(P_i - Q_i)^2}{P_i + Q_i + \\epsilon}$$
            - **Limitation**: **Zero spatial awareness**. A bag and a shirt with similar average brightness yield almost identical histograms.
            - **Typical Accuracy**: ~**25% – 32%** on test data.
            """
        )

# ==============================================================================
# TAB 2: COMPARE PERFORMANCE
# ==============================================================================
with tab2:
    st.subheader("Evaluation on Unseen Test Images")

    with st.spinner("Evaluating classifiers on test set..."):
        nn_result = evaluate_nearest_neighbor(nn_model, x_test, y_test)
        hist_result = evaluate_histogram(hist_templates, x_test, y_test)

    c_metric1, c_metric2 = st.columns(2)
    with c_metric1:
        st.metric(
            label="Nearest Neighbor Accuracy",
            value=f"{nn_result['accuracy'] * 100:.2f}%",
            delta=f"{(nn_result['accuracy'] - hist_result['accuracy']) * 100:+.2f}% vs Histogram",
        )
    with c_metric2:
        st.metric(
            label="Histogram Classifier Accuracy",
            value=f"{hist_result['accuracy'] * 100:.2f}%",
        )

    results_df = pd.DataFrame({
        "Classifier": ["1-Nearest Neighbor", "Histogram Template"],
        "Accuracy (%)": [f"{nn_result['accuracy'] * 100:.2f}%", f"{hist_result['accuracy'] * 100:.2f}%"],
        "Evaluation Time": [f"{nn_result['time']:.2f} s", f"{hist_result['time']:.2f} s"],
        "Spatial Information": ["Yes (Pixel alignment)", "No (Only pixel counts)"],
    })
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Confusion Matrices")
    st.caption("Rows represent actual ground-truth classes; columns represent predicted classes.")
    cm_c1, cm_c2 = st.columns(2)
    for col, result, title in [
        (cm_c1, nn_result, "1-Nearest Neighbor Confusion Matrix"),
        (cm_c2, hist_result, "Histogram Classifier Confusion Matrix"),
    ]:
        with col:
            fig, ax = plt.subplots(figsize=(6, 5))
            cax = ax.imshow(result["confusion_matrix"], cmap="Blues", interpolation="nearest")
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("True Class")
            ax.set_xticks(range(10))
            ax.set_yticks(range(10))
            ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(LABELS, fontsize=8)
            fig.colorbar(cax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("---")
    st.subheader("Example Test Predictions")
    example_count = min(8, len(x_test))
    ex_cols = st.columns(4)
    for i in range(example_count):
        nn_p = int(nn_result["predictions"][i])
        hist_p = int(hist_result["predictions"][i])
        actual_p = int(y_test[i])
        with ex_cols[i % 4]:
            st.image(x_test[i], clamp=True, width=100)
            status_nn = "✅" if nn_p == actual_p else "❌"
            status_h = "✅" if hist_p == actual_p else "❌"
            st.markdown(f"**Actual:** {LABELS[actual_p]}")
            st.markdown(f"**1-NN:** {LABELS[nn_p]} {status_nn}")
            st.markdown(f"**Hist:** {LABELS[hist_p]} {status_h}")

# ==============================================================================
# TAB 3: TEST CLOTHING PREDICTION
# ==============================================================================
with tab3:
    st.subheader("🎯 Test Clothing Identification")

    input_mode = st.radio(
        "Choose an image source to test:",
        ["📁 Upload custom photo / image", "🎯 Choose a pre-verified sample clothing item"],
        horizontal=True,
    )

    selected_image = None
    expected_label = None

    if input_mode == "📁 Upload custom photo / image":
        uploaded_file = st.file_uploader(
            "Upload an image of a clothing item (PNG, JPG, JPEG):",
            type=["png", "jpg", "jpeg"],
        )
        if uploaded_file is not None:
            selected_image = uploaded_file
    else:
        st.markdown("Select one of the 10 categories to load a verified item:")
        sample_folder = "sample_clothing"
        if os.path.exists(sample_folder):
            sample_files = sorted(os.listdir(sample_folder))
            sample_options = [f"{f.split('_')[0]}: {LABELS[int(f.split('_')[0])]}" for f in sample_files]
            choice = st.selectbox("Category sample:", sample_options)
            chosen_idx = int(choice.split(":")[0])
            chosen_file = sample_files[chosen_idx]
            selected_image = Image.open(os.path.join(sample_folder, chosen_file))
            expected_label = LABELS[chosen_idx]
            st.success(f"Loaded verified sample for: **{expected_label}**")

    if selected_image is not None:
        st.markdown("---")
        st.markdown("### 🛠️ Preprocessing & Background Controls")

        c_opt1, c_opt2, c_opt3 = st.columns(3)
        with c_opt1:
            invert_mode = st.selectbox(
                "Background Color Handling:",
                ["Auto (Detect light background and invert)", "Force Invert (White BG -> Black)", "Keep Original (Already Black BG)"],
                index=0,
            )
        with c_opt2:
            auto_center = st.checkbox("Auto-center & crop clothing item", value=True)
        with c_opt3:
            noise_threshold = st.slider("Shadow / Background noise filter", 0.0, 0.3, 0.05, 0.02)

        # Map UI choice to parameter
        if "Auto" in invert_mode:
            inv_param = "Auto"
        elif "Force" in invert_mode:
            inv_param = "Yes"
        else:
            inv_param = "No"

        # Run preprocessing
        processed_14x14, centered_28x28, was_inverted = preprocess_uploaded_image(
            selected_image,
            invert_background=inv_param,
            auto_center=auto_center,
            noise_threshold=noise_threshold,
        )

        st.markdown("---")
        st.markdown("### 👁️ Visual Diagnostic: What the Models See")

        c_view1, c_view2, c_view3 = st.columns(3)
        with c_view1:
            st.image(selected_image, caption="Original Uploaded Image", width=160)
            if expected_label:
                st.caption(f"Ground Truth Label: **{expected_label}**")
        with c_view2:
            st.image(centered_28x28, caption=f"Centered 28×28 Image (Inverted: {was_inverted})", width=160, clamp=True)
            st.caption("Matches Fashion-MNIST format (black background)")
        with c_view3:
            # Upscale 14x14 for clear display using nearest-neighbor
            display_14 = Image.fromarray((processed_14x14 * 255).astype(np.uint8)).resize((160, 160), Image.Resampling.NEAREST)
            st.image(display_14, caption="Final 14×14 Model Input (196 pixels)", width=160, clamp=True)

        # Run Predictions
        nn_pred, nn_dist, closest_train_img, closest_train_lbl = find_closest_training_example(
            nn_model, processed_14x14, full_x_train, full_y_train
        )
        hist_pred = predict_one_histogram(hist_templates, processed_14x14)
        hist_dists, img_hist = get_histogram_distances(hist_templates, processed_14x14)

        st.markdown("---")
        st.markdown("### 🏆 Prediction Results")

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.markdown(
                f"""
                <div style="background-color:#EFF6FF; border:2px solid #3B82F6; border-radius:10px; padding:1.2rem; text-align:center;">
                    <span style="background-color:#DBEAFE; color:#1D4ED8; font-weight:bold; padding:0.2rem 0.6rem; border-radius:6px;">1-Nearest Neighbor</span>
                    <h2 style="color:#1E40AF; margin:0.4rem 0;">{LABELS[nn_pred]}</h2>
                    <p style="margin:0; color:#374151;">Euclidean Distance to closest training item: <b>{nn_dist:.3f}</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with p_col2:
            st.markdown(
                f"""
                <div style="background-color:#FFF7ED; border:2px solid #F97316; border-radius:10px; padding:1.2rem; text-align:center;">
                    <span style="background-color:#FFEDD5; color:#C2410C; font-weight:bold; padding:0.2rem 0.6rem; border-radius:6px;">Histogram Classifier</span>
                    <h2 style="color:#9A3412; margin:0.4rem 0;">{LABELS[hist_pred]}</h2>
                    <p style="margin:0; color:#374151;">Chi-Square Distance to category template: <b>{hist_dists[hist_pred]:.3f}</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("#### 🔍 Explainability: Closest Training Image Matched by 1-NN")
        st.markdown(
            "Nearest Neighbor classifies by finding the single closest image in the dataset. "
            "Here is the exact training sample it matched with:"
        )

        c_exp1, c_exp2 = st.columns([1, 2])
        with c_exp1:
            closest_upscaled = Image.fromarray((closest_train_img * 255).astype(np.uint8)).resize((140, 140), Image.Resampling.NEAREST)
            st.image(closest_upscaled, caption=f"Matched: {LABELS[closest_train_lbl]} (dist: {nn_dist:.2f})", width=140, clamp=True)
        with c_exp2:
            st.markdown(
                f"""
                - **Matched Training Class**: `{LABELS[closest_train_lbl]}`
                - **Euclidean Distance**: `{nn_dist:.4f}`
                - **Why this matters**: If your uploaded image has clean, sharp silhouettes resembling this matched sample, 1-NN classifies it correctly!
                """
            )

        # Plot Chi-Square distances to all 10 categories
        st.markdown("#### 📊 Histogram Classifier: Distances to All 10 Categories")
        st.caption("Lower distance indicates a closer match to that category's average intensity distribution.")
        fig_d, ax_d = plt.subplots(figsize=(10, 3.5))
        bar_colors = ["#F97316" if i == hist_pred else "#CBD5E1" for i in range(10)]
        ax_d.bar(LABELS, hist_dists, color=bar_colors, edgecolor="black", linewidth=0.8)
        ax_d.set_ylabel("Chi-Square Distance", fontsize=10)
        ax_d.set_title("Histogram Distance per Clothing Class (Orange = Selected Winner)", fontsize=11, fontweight="bold")
        ax_d.set_xticklabels(LABELS, rotation=35, ha="right", fontsize=9)
        ax_d.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig_d)
        plt.close(fig_d)

st.divider()
st.caption("DS307 Computer Vision | Assignment 2 | Nearest Neighbor vs Histogram Classifier")
