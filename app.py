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
# CONFIGURATION
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
# PROFESSIONAL UI STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #f4f6f9;
    }

    /* Main width */
    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Header */
    .main-header {
        background-color: #102a43;
        padding: 22px 28px;
        border-radius: 14px;
        margin-bottom: 20px;
        color: white;
        box-shadow: 0 5px 18px rgba(0, 0, 0, 0.12);
    }

    .header-title {
        font-size: 30px;
        font-weight: 800;
        margin: 0;
    }

    .header-subtitle {
        color: #d8e3ed;
        font-size: 14px;
        margin-top: 5px;
    }

    /* Hero */
    .hero-box {
        background: linear-gradient(
            110deg,
            #0c263f 0%,
            #173d60 55%,
            #31546a 100%
        );
        padding: 38px 42px;
        border-radius: 15px;
        margin-bottom: 25px;
        color: white;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.13);
    }

    .hero-small {
        display: inline-block;
        background-color: #e52b32;
        padding: 6px 12px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 0.5px;
    }

    .hero-title {
        font-size: 40px;
        font-weight: 850;
        margin-top: 12px;
        margin-bottom: 8px;
    }

    .hero-description {
        color: #e1e9f0;
        font-size: 16px;
        max-width: 680px;
        line-height: 1.55;
    }

    /* Section title */
    .section-title {
        font-size: 23px;
        font-weight: 800;
        color: #152238;
        margin-top: 5px;
        margin-bottom: 3px;
    }

    .section-text {
        color: #68788a;
        font-size: 14px;
        margin-bottom: 15px;
    }

    /* Cards */
    .card {
        background-color: white;
        border: 1px solid #dce3ea;
        border-radius: 13px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(20, 42, 66, 0.06);
    }

    .card-title {
        color: #17385d;
        font-size: 19px;
        font-weight: 800;
        margin-bottom: 12px;
    }

    /* Result heading */
    .result-title {
        color: #152238;
        font-size: 25px;
        font-weight: 850;
    }

    .result-subtitle {
        color: #64748b;
        font-size: 14px;
        line-height: 1.5;
    }

    /* Metric styling */
    [data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #dce3ea;
        border-radius: 10px;
        padding: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: #64748b !important;
    }

    [data-testid="stMetricValue"] {
        color: #152238 !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #e52b32;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 800;
        min-height: 45px;
    }

    .stButton > button:hover {
        background-color: #c92128;
        color: white;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #f8fafc;
        border-radius: 10px;
    }

    /* Footer */
    .footer-box {
        background-color: #102a43;
        color: #dbe5ee;
        text-align: center;
        padding: 17px;
        border-radius: 12px;
        margin-top: 30px;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns([5, 1])

with header_left:
    st.markdown(
        """
        <div class="main-header">
            <div class="header-title">
                🚦 Traffic Sign Recognition System
            </div>
            <div class="header-subtitle">
                Road sign identification and classification system
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with header_right:
    st.success("● System Online")


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero-box">

        <div class="hero-small">
            TRAFFIC SAFETY SYSTEM
        </div>

        <div class="hero-title">
            Traffic Sign Inspection
        </div>

        <div class="hero-description">
            Upload a traffic sign image and identify its classification
            using a trained ResNet18 recognition model with unknown-sign
            detection.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():

    model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=MODEL_FILE,
        repo_type="model"
    )

    model = resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        43
    )

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
# PROTOTYPES
# ============================================================

@st.cache_data(show_spinner=False)
def load_prototypes():

    if not os.path.exists(PROTOTYPE_FILE):
        return None

    data = np.load(PROTOTYPE_FILE)

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

    x = F.normalize(
        x,
        p=2,
        dim=1
    )

    return x


# ============================================================
# PREDICTION
# ============================================================

def predict(model, image, prototypes):

    image_tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

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

    confidence = (
        confidence.item() * 100
    )

    top_probs = (
        top_probs[0]
        .cpu()
        .numpy() * 100
    )

    top_classes = (
        top_classes[0]
        .cpu()
        .numpy()
    )

    similarity = 0.0

    closest_class = predicted_class

    if prototypes is not None:

        feature_np = (
            feature[0]
            .cpu()
            .numpy()
        )

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
# MAIN SECTION
# ============================================================

left, right = st.columns(
    [0.95, 1.05],
    gap="large"
)


# ============================================================
# LEFT COLUMN - UPLOAD
# ============================================================

with left:

    st.markdown(
        '<div class="section-title">Traffic Sign Inspection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-text">'
        'Upload a clear image of a road traffic sign.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="card-title">📤 Upload Traffic Sign</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Select image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    image = None

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.write("")

        st.image(
            image,
            caption="Uploaded Traffic Sign",
            use_container_width=True
        )

        st.write("")

        analyze_button = st.button(
            "🔍 Analyze Traffic Sign",
            use_container_width=True
        )

    else:

        st.info(
            "📷 Upload a JPG, JPEG, or PNG image "
            "to begin traffic sign inspection."
        )

        analyze_button = False

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# RIGHT COLUMN - RESULT
# ============================================================

with right:

    st.markdown(
        '<div class="section-title">Detection Result</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-text">'
        'Classification and recognition analysis.'
        '</div>',
        unsafe_allow_html=True
    )

    if uploaded_file is None:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.info(
            "🔎 Waiting for an image. "
            "Upload a traffic sign on the left."
        )

        st.markdown(
            """
            **Recognition workflow**

            1. Upload a traffic sign image
            2. Analyze the image
            3. Compare model predictions
            4. Check feature similarity
            5. Return a recognized or unknown result
            """
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    elif not analyze_button:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.info(
            "Image uploaded successfully. "
            "Click **Analyze Traffic Sign** to start."
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    else:

        try:

            with st.spinner(
                "Analyzing traffic sign..."
            ):

                model = load_model()

                prototypes = load_prototypes()

                result = predict(
                    model,
                    image,
                    prototypes
                )

            predicted_class = result[
                "predicted_class"
            ]

            confidence = result[
                "confidence"
            ]

            similarity = result[
                "similarity"
            ]

            closest_class = result[
                "closest_class"
            ]

            top_classes = result[
                "top_classes"
            ]

            top_probs = result[
                "top_probs"
            ]

            is_known = result[
                "is_known"
            ]

            # ------------------------------------------------
            # KNOWN SIGN
            # ------------------------------------------------

            if is_known:

                st.success(
                    "✓ RECOGNIZED TRAFFIC SIGN"
                )

                st.markdown(
                    f"""
                    <div class="card">

                    <div class="result-title">
                        {CLASS_NAMES[predicted_class]}
                    </div>

                    <div class="result-subtitle">
                        The uploaded sign matches one of the
                        supported GTSRB traffic sign classes.
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ------------------------------------------------
            # UNKNOWN SIGN
            # ------------------------------------------------

            else:

                closest_name = CLASS_NAMES[
                    closest_class
                ]

                st.warning(
                    "⚠ UNKNOWN TRAFFIC SIGN"
                )

                st.markdown(
                    f"""
                    <div class="card">

                    <div class="result-title">
                        Sign Not Recognized
                    </div>

                    <div class="result-subtitle">

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

            st.write("")

            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            m1, m2 = st.columns(2)

            with m1:

                st.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%",
                    f"Required {CONFIDENCE_THRESHOLD:.0f}%"
                )

            with m2:

                st.metric(
                    "Feature Similarity",
                    f"{similarity:.4f}",
                    f"Required {SIMILARITY_THRESHOLD:.4f}"
                )

            st.write("")

            # ------------------------------------------------
            # TOP 5 PREDICTIONS
            # ------------------------------------------------

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="card-title">'
                '🏆 Top 5 Predictions'
                '</div>',
                unsafe_allow_html=True
            )

            for i in range(5):

                class_id = int(
                    top_classes[i]
                )

                probability = float(
                    top_probs[i]
                )

                name = CLASS_NAMES[
                    class_id
                ]

                st.write(
                    f"**{i + 1}. {name}**"
                )

                st.progress(
                    min(
                        probability / 100,
                        1.0
                    ),
                    text=f"{probability:.2f}%"
                )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

        except Exception as error:

            st.error(
                "An error occurred while analyzing the image."
            )

            with st.expander(
                "Technical details"
            ):
                st.code(
                    str(error)
                )


# ============================================================
# SYSTEM INFORMATION
# ============================================================

st.write("")
st.write("")

st.markdown(
    '<div class="section-title">System Information</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-text">'
    'Technical information about the recognition system.'
    '</div>',
    unsafe_allow_html=True
)

info1, info2, info3, info4 = st.columns(4)

with info1:

    st.metric(
        "Model",
        "ResNet18"
    )

with info2:

    st.metric(
        "Dataset",
        "GTSRB"
    )

with info3:

    st.metric(
        "Classes",
        "43"
    )

with info4:

    st.metric(
        "Status",
        "Ready"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-box">

        <strong>Traffic Sign Recognition System</strong>

        &nbsp; | &nbsp;

        Safer Roads. Smarter Technology.

    </div>
    """,
    unsafe_allow_html=True
)
