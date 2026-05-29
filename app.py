import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Set page config
st.set_page_config(page_title="LUNG CANCER DETECTOR", layout="wide", page_icon="🫁")

# --- FIXED HIGH-CONTRAST CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF !important; }
    .stMarkdown, p, span, label { color: #002D62 !important; }
    [data-testid="stSidebar"] { background-color: #002D62 !important; }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] h1 { color: #FFFFFF !important; }
    h1, h2, h3 { color: #004085 !important; font-family: 'Segoe UI', sans-serif; }
    .metric-card {
        background-color: #F8F9FA;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #DEE2E6;
        border-left: 8px solid #0056b3;
        margin-bottom: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    .metric-value { font-size: 30px; font-weight: bold; color: #155724 !important; }
    .metric-label { font-size: 16px; color: #495057 !important; font-weight: 600; }
    div[data-baseweb="select"] > div { background-color: #FFFFFF !important; color: #002D62 !important; }

    /* TARGETING UPLOAD TEXTS TO BE WHITE */
    /* This creates a blue box so the white text is visible */
    [data-testid="stFileUploader"] {
        background-color: #002D62;
        padding: 20px;
        border-radius: 10px;
    }
    [data-testid="stFileUploader"] label, [data-testid="stFileUploader"] p, [data-testid="stFileUploader"] small {
        color: #FFFFFF !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("# 🫁 MENU")
    st.divider()
    selection = st.selectbox("Navigation", ["🏠 Home", "🔬 REAL_TIME_SIMULATION", "📊 Performance Analysis", "👥 Research Team"])
    st.sidebar.markdown("---")
    st.sidebar.write("System Status: Online 🟢")

# Load Model
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('lung_model.h5')

# --- PAGE 1: HOME ---
if selection == "🏠 Home":
    st.title("🫁 LUNG CANCER DETECTOR")
    st.divider()

    st.markdown("""
    ### CNN-Based Diagnostic Architecture
    This system utilizes a **Deep Convolutional Neural Network (CNN)** specifically optimized for medical imaging.
    Unlike traditional algorithms, our CNN architecture automatically extracts spatial hierarchies of features
    from raw CT scans through multiple layers of **Convolution**, **Pooling**, and **Non-linear Activation**.

    **Technical Methodology:**
    - **Feature Extraction:** Convolutional filters detect micro-calcifications and structural irregularities in lung tissue.
    - **Transfer Learning:** Leveraging the **MobileNetV2** backbone for high-efficiency depthwise separable convolutions.
    - **Pattern Recognition:** The model identifies complex spatial patterns associated with malignancy that are often subtle to the human eye.

    **Core Performance:**
    - **97% Accuracy:** Validated on diverse clinical datasets.
    - **99% Sensitivity:** Tuned to minimize False Negatives for maximum patient safety.
    """)

# --- PAGE 2: REAL_TIME_SIMULATION ---
elif selection == "🔬 REAL_TIME_SIMULATION":
    st.title("🔬 REAL_TIME_SIMULATION")
    st.write("Upload a lung scan image (JPG/PNG) for neural classification.")

    # Text here will now be white inside a blue box
    uploaded_file = st.file_uploader("Upload Scan", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        model = load_my_model()
        image = Image.open(uploaded_file).convert('RGB')

        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(image, caption='Scan Input', use_container_width=True)

        with c2:
            st.markdown("### AI Diagnostic Output")
            img = image.resize((160, 160))
            img_array = np.array(img) / 255.0
            prediction = model.predict(np.expand_dims(img_array, axis=0))[0][0]

            st.divider()
            if prediction > 0.5:
                st.error("### Result: **CANCEROUS**")
                st.write(f"**Confidence:** {prediction*100:.2f}%")
                st.progress(int(prediction * 100))
            else:
                st.success("### Result: **NON-CANCEROUS**")
                st.write(f"**Confidence:** {(1-prediction)*100:.2f}%")
                st.progress(int((1 - prediction) * 100))

# --- PAGE 3: PERFORMANCE ANALYSIS ---
elif selection == "📊 Performance Analysis":
    st.title("📊 Statistical Validation")

    col_a, col_b = st.columns([1, 1.2])
    with col_a:
        st.markdown('<div class="metric-card"><div class="metric-label">OVERALL ACCURACY</div><div class="metric-value">97.0%</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card"><div class="metric-label">RECALL (SENSITIVITY)</div><div class="metric-value">99.0%</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card"><div class="metric-label">PRECISION</div><div class="metric-value">95.0%</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card"><div class="metric-label">F1-SCORE</div><div class="metric-value">97.0%</div></div>', unsafe_allow_html=True)

    with col_b:
        st.subheader("Confusion Matrix")
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", use_container_width=True)

        st.divider()
        st.markdown("""
        **Validation Summary:**
        - Tested on **219 clinical images**.
        - Correct Cancer Detection: **111 cases**.
        - Correct Non-Cancer Detection: **101 cases**.
        """)

# --- PAGE 4: RESEARCH TEAM ---
elif selection == "👥 Research Team":
    st.title("👥 Team & Mentorship")
    st.divider()

    st.markdown("### 🎓 Project Mentor")
    st.info("#### **SUBHRANGSHU DAS**")

    st.divider()
    st.markdown("### 👨‍💻 Research Team")
    team = ["**ANSH VERMA**", "**ARGHYA PARAMANIK**", "**BISHAL DUTTA**", "**ADRIJA SARKAR**", "**RANJINI MAJUMDER**"]
    for m in team:
        st.markdown(f"🔹 {m}")
