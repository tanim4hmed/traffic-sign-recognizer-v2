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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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
# PROFESSIONAL CSS
# ============================================================

st.html(
    """
    <style>

    /* ========================================================
       PAGE
       ======================================================== */

    .stApp {
        background-color: #f3f5f8;
    }

    /*
       IMPORTANT:
       Extra top padding keeps the application content
       safely below Streamlit's top toolbar.
    */

    .block-container {
        max-width: 1400px !important;
        padding-top: 6rem !important;
        padding-bottom: 2rem !important;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* ========================================================
       HEADER
       ======================================================== */

    .ts-header {
        width: 100%;
        box-sizing: border-box;
        background-color: #102a43;
        border-radius: 14px;
        padding: 24px 28px;
        margin-top: 4px;
        margin-bottom: 22px;
        min-height: 100px;
        box-shadow: 0 6px 20px rgba(16, 42, 67, 0.15);
    }

    .ts-header-title {
        color: #ffffff;
        font-size: 28px;
        font-weight: 800;
        line-height: 1.35;
        margin: 0;
        padding: 0;
    }

    .ts-header-subtitle {
        color: #d8e3ed;
        font-size: 14px;
        line-height: 1.4;
        margin-top: 7px;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .ts-hero {
        width: 100%;
        box-sizing: border-box;
        background: linear-gradient(
            115deg,
            #102a43 0%,
            #173f63 55%,
            #315c70 100%
        );
        border-radius: 15px;
        padding: 38px 40px;
        margin-bottom: 26px;
        box-shadow: 0 7px 22px rgba(16, 42, 67, 0.14);
    }

    .ts-hero-label {
        display: inline-block;
        background-color: #df2d35;
        color: #ffffff;
        padding: 7px 12px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.7px;
    }

    .ts-hero-title {
        color: #ffffff;
        font-size: 40px;
        font-weight: 850;
        line-height: 1.15;
        margin-top: 14px;
    }

    .ts-hero-text {
        color: #e4ebf1;
        font-size: 15px;
        line-height: 1.6;
        max-width: 720px;
        margin-top: 9px;
    }


    /* ========================================================
       SECTION TITLES
       ======================================================== */

    .ts-section-title {
        color: #152238;
        font-size: 22px;
        font-weight: 800;
        line-height: 1.3;
        margin-bottom: 4px;
    }

    .ts-section-text {
        color: #69798b;
        font-size: 13px;
        line-height: 1.5;
        margin-bottom: 14px;
    }


    /* ========================================================
       CARDS
       ======================================================== */

    .ts-card {
        background-color: #ffffff;
        border: 1px solid #dce3ea;
        border-radius: 13px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(20, 42, 66, 0.06);
        margin-bottom: 14px;
    }

    .ts-card-title {
        color: #17385d;
        font-size: 18px;
        font-weight: 800;
        line-height: 1.3;
        margin-bottom: 10px;
    }


    /* ========================================================
       RESULT
       ======================================================== */

    .ts-known {
        background-color: #eef9f2;
        border: 1px solid #c6e8d1;
        border-left: 5px solid #22a45a;
        border-radius: 10px;
        padding: 16px 17px;
        margin-bottom: 16px;
    }

    .ts-unknown {
        background-color: #fff1f1;
        border: 1px solid #f0c7c9;
        border-left: 5px solid #df2d35;
        border-radius: 10px;
        padding: 16px 17px;
        margin-bottom: 16px;
    }

    .ts-status-known {
        color: #168045;
        font-size: 11px;
        font-weight: 850;
        letter-spacing: 0.7px;
        margin-bottom: 6px;
    }

    .ts-status-unknown {
        color: #c5232b;
        font-size: 11px;
        font-weight: 850;
        letter-spacing: 0.7px;
        margin-bottom: 6px;
    }

    .ts-result-name {
        color: #152238;
        font-size: 25px;
        font-weight: 850;
        line-height: 1.25;
    }

    .ts-result-description {
        color: #657588;
        font-size: 13px;
        line-height: 1.55;
        margin-top: 7px;
    }


    /* ========================================================
       METRICS
       ======================================================== */

    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #dce3ea;
        border-radius: 11px;
        padding: 15px;
        box-shadow: 0 3px 10px rgba(20, 42, 66, 0.04);
    }

    [data-testid="stMetricLabel"] {
        color: #64748b !important;
    }

    [data-testid="stMetricValue"] {
        color: #152238 !important;
    }


    /* ========================================================
       BUTTON
       ======================================================== */

    .stButton > button {
        width: 100%;
        background-color: #df2d35;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        min-height: 46px;
        font-size: 15px;
        font-weight: 800;
        transition: 0.15s ease;
    }

    .stButton > button:hover {
        background-color: #c9232b;
        color: #ffffff;
    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    [data-testid="stFileUploader"] {
        background-color: #f8fafc;
        border: 1px solid #dce3ea;
        border-radius: 10px;
        padding: 6px;
    }


    /* ========================================================
       PROGRESS BAR
       ======================================================== */

    [data-testid="stProgressBar"] > div > div > div {
        background-color: #2469c8;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .ts-footer {
        background-color: #102a43;
        color: #dbe5ee;
        text-align: center;
        padding: 17px;
        border-radius: 12px;
        margin-top: 30px;
        font-size: 12px;
    }

    .ts-footer strong {
        color: #ffffff;
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 800px) {

        .block-container {
            padding-top: 5rem !important;
        }

        .ts-header {
            padding: 21px;
        }

        .ts-header-title {
            font-size: 22px;
        }

        .ts-hero {
            padding: 28px 24px;
        }

        .ts-hero-title {
            font-size: 30px;
        }

        .ts-hero-text {
            font-size: 14px;
        }
    }

    </style>
    """
)


# ============================================================
# HEADER
# ============================================================

st.html(
    """
    <div class="ts-header">

        <div class="ts-header-title">
            🚦 Traffic Sign Recognition System
        </div>

        <div class="ts-header-subtitle">
            Road sign identification and classification system
        </div>

    </div>
    """
)


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="ts-hero">

        <div class="ts-hero-label">
            TRAFFIC SAFETY SYSTEM
        </div>

        <div class="ts-hero-title">
            Traffic Sign Inspection
        </div>

        <div class="ts-hero-text">
            Upload a traffic sign image and identify its
            classification using a trained ResNet18 recognition
            model with unknown-sign detection.
        </div>

    </div>
    """
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
# PROTOTYPE LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_prototypes():

    if not os.path.exists(PROTOTYPE_FILE):
        return None

    data = np.load(
        PROTOTYPE_FILE
    )

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

def extract_feature(
    model,
    image_tensor
):

    x = model.conv1(
        image_tensor
    )

    x = model.bn1(x)

    x = model.relu(x)

    x = model.maxpool(x)

    x = model.layer1(x)

    x = model.layer2(x)

    x = model.layer3(x)

    x = model.layer4(x)

    x = model.avgpool(x)

    x = torch.flatten(
        x,
        1
    )

    x = F.normalize(
        x,
        p=2,
        dim=1
    )

    return x


# ============================================================
# PREDICTION
# ============================================================

def predict(
    model,
    image,
    prototypes
):

    image_tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

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

    predicted_class = (
        predicted_class.item()
    )

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
            np.argmax(
                similarities
            )
        )

        similarity = float(
            similarities[
                closest_class
            ]
        )

    is_known = (
        confidence >= CONFIDENCE_THRESHOLD
        and
        similarity >= SIMILARITY_THRESHOLD
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
# MAIN WORKSPACE
# ============================================================

left_column, right_column = st.columns(
    [0.95, 1.05],
    gap="large"
)


# ============================================================
# LEFT COLUMN
# ============================================================

with left_column:

    st.html(
        """
        <div class="ts-section-title">
            Traffic Sign Inspection
        </div>

        <div class="ts-section-text">
            Upload a clear road-sign image for classification.
        </div>
        """
    )

    st.html(
        """
        <div class="ts-card">

            <div class="ts-card-title">
                📤 Upload Traffic Sign
            </div>

        </div>
        """
    )

    uploaded_file = st.file_uploader(
        "Choose a traffic sign image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        label_visibility="collapsed"
    )

    image = None

    analyze_button = False

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Traffic Sign",
            use_container_width=True
        )

        analyze_button = st.button(
            "🔍 Analyze Traffic Sign",
            use_container_width=True
        )

    else:

        st.info(
            "📷 Upload a JPG, JPEG, or PNG traffic sign image."
        )


# ============================================================
# RIGHT COLUMN
# ============================================================

with right_column:

    st.html(
        """
        <div class="ts-section-title">
            Detection Result
        </div>

        <div class="ts-section-text">
            Classification and recognition analysis.
        </div>
        """
    )

    if uploaded_file is None:

        st.html(
            """
            <div class="ts-card">

                <div class="ts-card-title">
                    🔎 Waiting for Image
                </div>

                <div>
                    Upload a traffic sign image on the left
                    to begin the inspection.
                </div>

            </div>
            """
        )

    elif not analyze_button:

        st.info(
            "Image uploaded successfully. "
            "Click 'Analyze Traffic Sign' to continue."
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


            # ==================================================
            # RECOGNIZED SIGN
            # ==================================================

            if is_known:

                st.html(
                    f"""
                    <div class="ts-known">

                        <div class="ts-status-known">
                            ✓ RECOGNIZED TRAFFIC SIGN
                        </div>

                        <div class="ts-result-name">
                            {CLASS_NAMES[predicted_class]}
                        </div>

                        <div class="ts-result-description">
                            The uploaded sign matches one of
                            the supported GTSRB traffic sign
                            classes.
                        </div>

                    </div>
                    """
                )


            # ==================================================
            # UNKNOWN SIGN
            # ==================================================

            else:

                closest_name = CLASS_NAMES[
                    closest_class
                ]

                st.html(
                    f"""
                    <div class="ts-unknown">

                        <div class="ts-status-unknown">
                            ⚠ UNKNOWN TRAFFIC SIGN
                        </div>

                        <div class="ts-result-name">
                            Sign Not Recognized
                        </div>

                        <div class="ts-result-description">
                            This sign could not be confidently
                            matched with one of the 43 supported
                            GTSRB traffic sign classes.
                            <br><br>
                            Highest model prediction:
                            <strong>{closest_name}</strong>
                        </div>

                    </div>
                    """
                )


            # ==================================================
            # METRICS
            # ==================================================

            metric_left, metric_right = st.columns(2)

            with metric_left:

                st.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%",
                    f"Threshold: {CONFIDENCE_THRESHOLD:.0f}%"
                )

            with metric_right:

                st.metric(
                    "Feature Similarity",
                    f"{similarity:.4f}",
                    f"Threshold: {SIMILARITY_THRESHOLD:.4f}"
                )


            # ==================================================
            # UNKNOWN DECISION DETAILS
            # ==================================================

            if not is_known:

                st.warning(
                    f"Final decision: NOT RECOGNIZED — "
                    f"feature similarity {similarity:.4f} "
                    f"is below the required "
                    f"{SIMILARITY_THRESHOLD:.4f} threshold."
                )


            # ==================================================
            # TOP 5 PREDICTIONS
            # ==================================================

            st.html(
                """
                <div class="ts-card">

                    <div class="ts-card-title">
                        🏆 Top 5 Predictions
                    </div>

                </div>
                """
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
                    f"**{i + 1}. {name}** — "
                    f"{probability:.2f}%"
                )

                st.progress(
                    min(
                        probability / 100,
                        1.0
                    )
                )


        except Exception as error:

            st.error(
                "Unable to analyze the image."
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

st.html(
    """
    <div class="ts-section-title">
        System Information
    </div>

    <div class="ts-section-text">
        Technical information about the recognition system.
    </div>
    """
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
        "Supported Classes",
        "43"
    )

with info4:

    st.metric(
        "System Status",
        "Ready"
    )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="ts-footer">

        <strong>
            Traffic Sign Recognition System
        </strong>

        &nbsp; | &nbsp;

        Safer Roads. Smarter Technology.

    </div>
    """
)
