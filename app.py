import os
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
from huggingface_hub import hf_hub_download


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Traffic Sign Recognition System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CONSTANTS
# ============================================================

HF_REPO = "tanim4hmed/traffic-sign-recognizer-v2"
MODEL_FILE = "traffic_sign_resnet18_final.pth"
PROTOTYPE_FILE = "traffic_sign_prototypes.npz"

CONFIDENCE_THRESHOLD = 70.0
SIMILARITY_THRESHOLD = 0.90

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# GTSRB CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Speed limit (20 km/h)",
    "Speed limit (30 km/h)",
    "Speed limit (50 km/h)",
    "Speed limit (60 km/h)",
    "Speed limit (70 km/h)",
    "Speed limit (80 km/h)",
    "End of speed limit (80 km/h)",
    "Speed limit (100 km/h)",
    "Speed limit (120 km/h)",
    "No passing",
    "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 metric tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons"
]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ------------------------------
       GLOBAL
    ------------------------------ */

    .stApp {
        background: #f4f6f9;
        color: #152238;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* Hide Streamlit default menu */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* ------------------------------
       HEADER
    ------------------------------ */

    .site-header {
        background: #102a43;
        border-radius: 0 0 16px 16px;
        padding: 20px 28px;
        margin-bottom: 18px;
        color: white;
        box-shadow: 0 5px 18px rgba(15, 35, 60, 0.15);
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 18px;
    }

    .brand-icon {
        width: 62px;
        height: 62px;
        background: #e52b32;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 35px;
        flex-shrink: 0;
    }

    .brand-title {
        font-size: 28px;
        font-weight: 800;
        line-height: 1.1;
        margin: 0;
    }

    .brand-subtitle {
        font-size: 14px;
        color: #d7e2ee;
        margin-top: 7px;
    }

    .online-status {
        margin-left: auto;
        background: rgba(31, 170, 89, 0.18);
        border: 1px solid rgba(90, 220, 130, 0.35);
        color: #63df91;
        border-radius: 30px;
        padding: 8px 15px;
        font-size: 13px;
        font-weight: 700;
        white-space: nowrap;
    }

    .online-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #35d779;
        border-radius: 50%;
        margin-right: 7px;
    }


    /* ------------------------------
       HERO
    ------------------------------ */

    .hero {
        position: relative;
        overflow: hidden;
        min-height: 250px;
        border-radius: 16px;
        margin-bottom: 22px;
        background:
            linear-gradient(
                90deg,
                rgba(7, 25, 44, 0.96) 0%,
                rgba(7, 25, 44, 0.86) 42%,
                rgba(7, 25, 44, 0.30) 100%
            ),
            linear-gradient(
                135deg,
                #173d60 0%,
                #294f69 50%,
                #7a8d91 100%
            );
        box-shadow: 0 7px 22px rgba(20, 42, 66, 0.15);
    }

    .hero-road {
        position: absolute;
        right: -30px;
        bottom: -90px;
        width: 60%;
        height: 220px;
        transform: rotate(-4deg);
        background:
            linear-gradient(
                90deg,
                transparent 0%,
                rgba(255,255,255,0.8) 48%,
                transparent 50%
            );
        opacity: 0.18;
    }

    .hero-content {
        position: relative;
        z-index: 2;
        padding: 38px 42px;
        max-width: 700px;
    }

    .hero-label {
        display: inline-block;
        background: #e52b32;
        color: white;
        padding: 7px 12px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 0.8px;
        margin-bottom: 13px;
    }

    .hero-title {
        color: white;
        font-size: 43px;
        font-weight: 850;
        margin: 0;
        line-height: 1.05;
    }

    .hero-text {
        color: #e6edf4;
        font-size: 16px;
        margin-top: 14px;
        line-height: 1.55;
        max-width: 600px;
    }


    /* ------------------------------
       SECTION TITLES
    ------------------------------ */

    .section-heading {
        color: #152238;
        font-size: 22px;
        font-weight: 800;
        margin: 6px 0 4px 0;
    }

    .section-description {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 15px;
    }


    /* ------------------------------
       CARDS
    ------------------------------ */

    .panel {
        background: white;
        border: 1px solid #dce3ea;
        border-radius: 14px;
        box-shadow: 0 4px 15px rgba(20, 42, 66, 0.07);
        overflow: hidden;
    }

    .panel-header {
        background: #17385d;
        color: white;
        padding: 15px 19px;
        font-size: 18px;
        font-weight: 750;
    }

    .panel-body {
        padding: 20px;
    }


    /* ------------------------------
       UPLOAD AREA
    ------------------------------ */

    .upload-title {
        color: #152238;
        font-size: 20px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .upload-subtitle {
        color: #718096;
        font-size: 13px;
        margin-bottom: 18px;
    }

    [data-testid="stFileUploader"] {
        background: #f8fafc;
        border: 2px dashed #b7c5d5;
        border-radius: 12px;
        padding: 8px;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #e52b32;
    }

    .image-caption {
        color: #43556b;
        font-size: 14px;
        font-weight: 750;
        margin-top: 17px;
        margin-bottom: 8px;
    }


    /* ------------------------------
       RESULT STATUS
    ------------------------------ */

    .status-known {
        background: #eaf8ef;
        border: 1px solid #bde7ca;
        border-left: 5px solid #21a45a;
        border-radius: 10px;
        padding: 17px 18px;
        margin-bottom: 18px;
    }

    .status-unknown {
        background: #fff1f1;
        border: 1px solid #f2c5c7;
        border-left: 5px solid #e52b32;
        border-radius: 10px;
        padding: 17px 18px;
        margin-bottom: 18px;
    }

    .status-label {
        font-size: 12px;
        font-weight: 850;
        letter-spacing: 0.7px;
        margin-bottom: 5px;
    }

    .status-label-known {
        color: #168045;
    }

    .status-label-unknown {
        color: #c5232b;
    }

    .result-name {
        color: #152238;
        font-size: 27px;
        font-weight: 850;
        margin-top: 4px;
        margin-bottom: 5px;
    }

    .result-description {
        color: #5e6f82;
        font-size: 14px;
        line-height: 1.5;
    }


    /* ------------------------------
       METRICS
    ------------------------------ */

    .metric-card {
        background: #f8fafc;
        border: 1px solid #dbe3eb;
        border-radius: 11px;
        padding: 17px;
        height: 100%;
    }

    .metric-label {
        color: #617287;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }

    .metric-value {
        color: #152238;
        font-size: 27px;
        font-weight: 850;
        margin-top: 4px;
    }

    .metric-note {
        color: #7b8998;
        font-size: 11px;
        margin-top: 3px;
    }


    /* ------------------------------
       PREDICTIONS
    ------------------------------ */

    .prediction-row {
        margin-bottom: 14px;
    }

    .prediction-name {
        color: #27364a;
        font-size: 13px;
        font-weight: 650;
        margin-bottom: 5px;
    }

    .prediction-track {
        height: 9px;
        background: #e6ebf1;
        border-radius: 20px;
        overflow: hidden;
    }

    .prediction-fill {
        height: 100%;
        background: #2469c8;
        border-radius: 20px;
    }

    .prediction-fill-top {
        background: #e52b32;
    }

    .prediction-percent {
        color: #152238;
        font-size: 13px;
        font-weight: 800;
        text-align: right;
    }


    /* ------------------------------
       SYSTEM INFO
    ------------------------------ */

    .info-card {
        background: white;
        border: 1px solid #dce3ea;
        border-radius: 14px;
        padding: 19px;
        box-shadow: 0 4px 15px rgba(20, 42, 66, 0.06);
    }

    .info-label {
        color: #7a8898;
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .info-value {
        color: #152238;
        font-size: 16px;
        font-weight: 800;
        margin-top: 5px;
    }


    /* ------------------------------
       BUTTON
    ------------------------------ */

    .stButton > button {
        width: 100%;
        border-radius: 9px;
        border: none;
        background: #e52b32;
        color: white;
        font-size: 15px;
        font-weight: 800;
        padding: 12px 20px;
        min-height: 46px;
        box-shadow: 0 4px 10px rgba(229, 43, 50, 0.22);
        transition: all 0.18s ease;
    }

    .stButton > button:hover {
        background: #c92128;
        transform: translateY(-1px);
        box-shadow: 0 6px 14px rgba(229, 43, 50, 0.28);
    }


    /* ------------------------------
       FOOTER
    ------------------------------ */

    .footer {
        margin-top: 30px;
        background: #102a43;
        color: #dce7f1;
        border-radius: 13px;
        padding: 18px 24px;
        text-align: center;
        font-size: 13px;
    }

    .footer strong {
        color: white;
    }

    .footer-red {
        color: #f15b61;
        font-weight: 800;
    }


    /* ------------------------------
       MOBILE
    ------------------------------ */

    @media (max-width: 800px) {

        .brand-title {
            font-size: 21px;
        }

        .brand-subtitle {
            font-size: 12px;
        }

        .online-status {
            display: none;
        }

        .hero-content {
            padding: 30px 25px;
        }

        .hero-title {
            font-size: 31px;
        }

        .hero-text {
            font-size: 14px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="site-header">
        <div class="brand-row">
            <div class="brand-icon">🚦</div>

            <div>
                <div class="brand-title">
                    Traffic Sign Recognition System
                </div>

                <div class="brand-subtitle">
                    Road sign identification and classification system
                </div>
            </div>

            <div class="online-status">
                <span class="online-dot"></span>
                System Online
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-road"></div>

        <div class="hero-content">

            <div class="hero-label">
                TRAFFIC SAFETY SYSTEM
            </div>

            <div class="hero-title">
                Traffic Sign Inspection
            </div>

            <div class="hero-text">
                Upload a traffic sign image and identify its
                classification using our trained road-sign
                recognition model.
            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():

    model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=MODEL_FILE,
        repo_type="model"
    )

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 43)

    state_dict = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# PROTOTYPE LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_prototypes():

    path = PROTOTYPE_FILE

    if not os.path.exists(path):
        return None

    data = np.load(path)

    return data["prototypes"]


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_feature(model, image_tensor):

    x = model.conv1(image_tensor)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    x = torch.flatten(x, 1)

    x = F.normalize(x, p=2, dim=1)

    return x


# ============================================================
# PREDICTION
# ============================================================

def predict(model, image, prototypes):

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

        top_probs, top_classes = torch.topk(
            probabilities,
            5,
            dim=1
        )

        feature = extract_feature(
            model,
            image_tensor
        )

    predicted_class = predicted_class.item()
    confidence = confidence.item() * 100

    top_probs = top_probs[0].cpu().numpy() * 100
    top_classes = top_classes[0].cpu().numpy()

    # ----------------------------------------
    # Prototype similarity
    # ----------------------------------------

    similarity = 0.0
    closest_class = predicted_class

    if prototypes is not None:

        feature_np = feature[0].cpu().numpy()

        similarities = np.dot(
            prototypes,
            feature_np
        )

        closest_class = int(
            np.argmax(similarities)
        )

        similarity = float(
            similarities[closest_class]
        )

    # ----------------------------------------
    # Final decision
    # ----------------------------------------

    is_known = (
        confidence >= CONFIDENCE_THRESHOLD
        and similarity >= SIMILARITY_THRESHOLD
    )

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top_classes": top_classes,
        "top_probs": top_probs,
        "similarity": similarity,
        "closest_class": closest_class,
        "is_known": is_known
    }


# ============================================================
# MAIN CONTENT
# ============================================================

left_col, right_col = st.columns(
    [0.9, 1.1],
    gap="large"
)


# ============================================================
# LEFT: IMAGE UPLOAD
# ============================================================

with left_col:

    st.markdown(
        '<div class="section-heading">Traffic Sign Inspection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Upload a clear image of a road traffic sign for analysis.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="panel-header">📤 Upload Traffic Sign</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="panel-body">',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.markdown(
            '<div class="image-caption">Uploaded Image</div>',
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )

        st.markdown(
            "<div style='height:10px'></div>",
            unsafe_allow_html=True
        )

        recognize = st.button(
            "🔍 Analyze Traffic Sign",
            use_container_width=True
        )

    else:

        st.markdown(
            """
            <div style="
                text-align:center;
                padding:42px 15px 35px 15px;
                color:#6d7c8e;
            ">
                <div style="font-size:48px;">🛣️</div>

                <div style="
                    font-size:18px;
                    font-weight:800;
                    color:#30445c;
                    margin-top:10px;
                ">
                    Upload a traffic sign image
                </div>

                <div style="
                    font-size:13px;
                    margin-top:7px;
                ">
                    JPG, JPEG or PNG
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        recognize = False

    st.markdown(
        '</div></div>',
        unsafe_allow_html=True
    )


# ============================================================
# RIGHT: DETECTION RESULT
# ============================================================

with right_col:

    st.markdown(
        '<div class="section-heading">Detection Result</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Classification and recognition analysis will appear here.'
        '</div>',
        unsafe_allow_html=True
    )

    if uploaded_file is None:

        st.markdown(
            """
            <div class="panel">
                <div class="panel-header">
                    📊 Detection Result
                </div>

                <div class="panel-body">

                    <div style="
                        text-align:center;
                        padding:75px 20px;
                        color:#738398;
                    ">

                        <div style="
                            font-size:52px;
                            margin-bottom:12px;
                        ">
                            🔎
                        </div>

                        <div style="
                            color:#30445c;
                            font-size:19px;
                            font-weight:800;
                        ">
                            Waiting for image
                        </div>

                        <div style="
                            margin-top:8px;
                            font-size:13px;
                        ">
                            Upload a traffic sign to begin inspection.
                        </div>

                    </div>

                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif recognize:

        try:

            with st.spinner("Analyzing traffic sign..."):

                model = load_model()
                prototypes = load_prototypes()

                result = predict(
                    model,
                    image,
                    prototypes
                )

            predicted_class = result["predicted_class"]
            confidence = result["confidence"]
            similarity = result["similarity"]

            top_classes = result["top_classes"]
            top_probs = result["top_probs"]

            is_known = result["is_known"]

            # ----------------------------------------
            # RESULT PANEL
            # ----------------------------------------

            st.markdown(
                '<div class="panel">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="panel-header">'
                '📊 Detection Result'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="panel-body">',
                unsafe_allow_html=True
            )

            # ----------------------------------------
            # KNOWN RESULT
            # ----------------------------------------

            if is_known:

                st.markdown(
                    f"""
                    <div class="status-known">

                        <div class="status-label status-label-known">
                            ✓ RECOGNIZED TRAFFIC SIGN
                        </div>

                        <div class="result-name">
                            {CLASS_NAMES[predicted_class]}
                        </div>

                        <div class="result-description">
                            The uploaded sign matches one of the
                            supported GTSRB traffic sign classes.
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ----------------------------------------
            # UNKNOWN RESULT
            # ----------------------------------------

            else:

                closest_name = CLASS_NAMES[
                    result["closest_class"]
                ]

                st.markdown(
                    f"""
                    <div class="status-unknown">

                        <div class="status-label status-label-unknown">
                            ⚠ UNKNOWN TRAFFIC SIGN
                        </div>

                        <div class="result-name">
                            Sign Not Recognized
                        </div>

                        <div class="result-description">
                            This sign could not be confidently
                            matched with one of the 43 supported
                            GTSRB traffic sign classes.
                            <br><br>
                            The model's highest prediction was
                            <strong>{closest_name}</strong>,
                            but the feature similarity was below
                            the required recognition threshold.
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ----------------------------------------
            # METRICS
            # ----------------------------------------

            metric1, metric2 = st.columns(2)

            with metric1:

                st.markdown(
                    f"""
                    <div class="metric-card">

                        <div class="metric-label">
                            Model Confidence
                        </div>

                        <div class="metric-value">
                            {confidence:.2f}%
                        </div>

                        <div class="metric-note">
                            Required: {CONFIDENCE_THRESHOLD:.0f}%
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with metric2:

                st.markdown(
                    f"""
                    <div class="metric-card">

                        <div class="metric-label">
                            Feature Similarity
                        </div>

                        <div class="metric-value">
                            {similarity:.4f}
                        </div>

                        <div class="metric-note">
                            Required: {SIMILARITY_THRESHOLD:.4f}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(
                "<div style='height:22px'></div>",
                unsafe_allow_html=True
            )

            # ----------------------------------------
            # TOP 5
            # ----------------------------------------

            st.markdown(
                """
                <div style="
                    color:#152238;
                    font-size:19px;
                    font-weight:800;
                    margin-bottom:15px;
                ">
                    🏆 Top 5 Predictions
                </div>
                """,
                unsafe_allow_html=True
            )

            for i in range(5):

                class_id = int(top_classes[i])
                probability = float(top_probs[i])

                name = CLASS_NAMES[class_id]

                width = max(
                    min(probability, 100),
                    0
                )

                fill_class = (
                    "prediction-fill-top"
                    if i == 0
                    else "prediction-fill"
                )

                st.markdown(
                    f"""
                    <div class="prediction-row">

                        <div style="
                            display:flex;
                            justify-content:space-between;
                            align-items:center;
                            gap:10px;
                        ">

                            <div class="prediction-name">
                                {i + 1}. {name}
                            </div>

                            <div class="prediction-percent">
                                {probability:.2f}%
                            </div>

                        </div>

                        <div class="prediction-track">

                            <div
                                class="{fill_class}"
                                style="width:{width}%"
                            ></div>

                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(
                '</div></div>',
                unsafe_allow_html=True
            )

        except Exception as e:

            st.error(
                "An error occurred while analyzing the image."
            )

            st.code(str(e))

    else:

        st.markdown(
            """
            <div class="panel">
                <div class="panel-header">
                    📊 Detection Result
                </div>

                <div class="panel-body">

                    <div style="
                        text-align:center;
                        padding:55px 20px;
                        color:#738398;
                    ">

                        <div style="font-size:45px;">
                            ✓
                        </div>

                        <div style="
                            color:#30445c;
                            font-size:18px;
                            font-weight:800;
                            margin-top:10px;
                        ">
                            Image ready for analysis
                        </div>

                        <div style="
                            font-size:13px;
                            margin-top:7px;
                        ">
                            Click "Analyze Traffic Sign" to continue.
                        </div>

                    </div>

                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# SYSTEM INFORMATION
# ============================================================

st.markdown(
    "<div style='height:25px'></div>",
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-heading">System Information</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Technical information about the recognition system.'
    '</div>',
    unsafe_allow_html=True
)

info1, info2, info3, info4 = st.columns(4)

with info1:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-label">
                Model
            </div>

            <div class="info-value">
                ResNet18
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with info2:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-label">
                Dataset
            </div>

            <div class="info-value">
                GTSRB
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with info3:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-label">
                Supported Classes
            </div>

            <div class="info-value">
                43
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with info4:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-label">
                System Status
            </div>

            <div class="info-value"
                 style="color:#1c9a50;">
                ● Ready
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <strong>Traffic Sign Recognition System</strong>

        &nbsp;&nbsp;|&nbsp;&nbsp;

        <span class="footer-red">
            Safer Roads.
        </span>

        Smarter Technology.

    </div>
    """,
    unsafe_allow_html=True
)
