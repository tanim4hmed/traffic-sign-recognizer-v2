import os
import time

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
    page_title="Roadsign Reader",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
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
    "End of no passing by vehicles over 3.5 metric tons",
]


# ============================================================
# STYLE
# Visual language: real road signage.
#   highway green panels with a white inner rule,
#   warning-yellow for "unknown", asphalt text,
#   and a dashed centre line that keeps moving.
# ============================================================

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Barlow:wght@400;500;600&display=swap');

:root {
    --asphalt: #14181c;
    --asphalt-soft: #4a545e;
    --paper: #eceee9;
    --card: #ffffff;
    --hairline: #d5d9d2;
    --green: #0b5d3b;
    --green-deep: #08442b;
    --green-light: #e3f1e9;
    --yellow: #ffc400;
    --yellow-light: #fff6d1;
    --red: #c62828;
    --blue: #1f5fa8;
    --display: 'Barlow Condensed', 'Arial Narrow', 'Segoe UI', sans-serif;
    --body: 'Barlow', 'Segoe UI', Roboto, Arial, sans-serif;
}

/* ---------- page ---------- */

.stApp {
    background-color: var(--paper);
    font-family: var(--body);
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1320px !important;
    padding-top: 4.2rem !important;
    padding-bottom: 2rem !important;
}

#MainMenu, footer { visibility: hidden; }


/* ---------- masthead: a highway sign ---------- */

.rs-sign {
    position: relative;
    display: flex;
    align-items: center;
    gap: 34px;
    background: var(--green);
    color: #fff;
    border-radius: 22px;
    padding: 34px 44px;
    box-shadow:
        inset 0 0 0 5px var(--green),
        inset 0 0 0 8px rgba(255, 255, 255, 0.92),
        0 14px 34px rgba(8, 68, 43, 0.28);
}

.rs-diamond {
    flex: 0 0 auto;
    width: 74px;
    height: 74px;
    background: var(--yellow);
    border: 4px solid var(--asphalt);
    border-radius: 10px;
    transform: rotate(45deg);
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 12px;
}

.rs-diamond span {
    transform: rotate(-45deg);
    font-family: var(--display);
    font-weight: 800;
    font-size: 44px;
    color: var(--asphalt);
    line-height: 1;
}

.rs-sign-title {
    font-family: var(--display);
    font-weight: 800;
    font-size: 54px;
    line-height: 1;
    letter-spacing: 0.5px;
}

.rs-sign-sub {
    font-family: var(--body);
    font-size: 16px;
    line-height: 1.5;
    margin-top: 10px;
    max-width: 620px;
    color: #d6eadf;
}

.rs-sign-tags {
    margin-left: auto;
    display: flex;
    flex-direction: column;
    gap: 8px;
    align-items: flex-end;
}

.rs-tag {
    font-family: var(--display);
    font-weight: 700;
    font-size: 17px;
    background: rgba(255, 255, 255, 0.12);
    border: 1.5px solid rgba(255, 255, 255, 0.55);
    border-radius: 8px;
    padding: 4px 14px;
    white-space: nowrap;
}

/* the one moving thing on the page: a centre line */

.rs-road {
    height: 12px;
    margin: 26px 0 30px 0;
    border-radius: 6px;
    background-color: var(--asphalt);
    background-image: repeating-linear-gradient(
        90deg,
        var(--yellow) 0 46px,
        transparent 46px 86px
    );
    background-size: 86px 3px;
    background-repeat: repeat-x;
    background-position: 0 50%;
    animation: rs-drive 1.6s linear infinite;
}

@keyframes rs-drive {
    from { background-position: 0 50%; }
    to   { background-position: -86px 50%; }
}


/* ---------- section headings ---------- */

.rs-h2 {
    font-family: var(--display);
    font-weight: 700;
    font-size: 30px;
    line-height: 1.1;
    color: var(--asphalt);
    margin: 0 0 4px 0;
}

.rs-p {
    font-family: var(--body);
    font-size: 14.5px;
    line-height: 1.55;
    color: var(--asphalt-soft);
    margin: 0 0 16px 0;
}


/* ---------- uploader ---------- */

[data-testid="stFileUploaderDropzone"] {
    background-color: var(--yellow-light);
    border: 2.5px dashed #d9a800;
    border-radius: 16px;
    padding: 26px 20px;
    transition: background-color 0.15s ease, border-color 0.15s ease;
}

[data-testid="stFileUploaderDropzone"]:hover {
    background-color: #fff0b3;
    border-color: var(--asphalt);
}

[data-testid="stFileUploaderDropzone"] button {
    background-color: var(--asphalt);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}

[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span {
    color: var(--asphalt-soft);
}

[data-testid="stImage"] img {
    border-radius: 16px;
    border: 4px solid #fff;
    box-shadow: 0 10px 26px rgba(20, 24, 28, 0.18);
}

[data-testid="stImageCaption"] {
    color: var(--asphalt-soft);
}


/* ---------- verdict panels ---------- */

.rs-verdict {
    display: flex;
    align-items: center;
    gap: 26px;
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 18px;
    animation: rs-pop 0.5s cubic-bezier(0.2, 0.9, 0.3, 1.2) both;
}

.rs-verdict.known {
    background: var(--green);
    color: #fff;
    box-shadow:
        inset 0 0 0 4px var(--green),
        inset 0 0 0 7px rgba(255, 255, 255, 0.92),
        0 12px 28px rgba(8, 68, 43, 0.25);
}

.rs-verdict.unknown {
    background: var(--yellow);
    color: var(--asphalt);
    box-shadow:
        inset 0 0 0 4px var(--yellow),
        inset 0 0 0 7px var(--asphalt),
        0 12px 28px rgba(150, 110, 0, 0.28);
}

@keyframes rs-pop {
    from { opacity: 0; transform: scale(0.94); }
    to   { opacity: 1; transform: scale(1); }
}

.rs-verdict-body { flex: 1 1 auto; min-width: 0; }

.rs-verdict-kicker {
    font-family: var(--display);
    font-weight: 700;
    font-size: 18px;
    opacity: 0.85;
}

.rs-verdict-name {
    font-family: var(--display);
    font-weight: 800;
    font-size: 40px;
    line-height: 1.05;
    margin-top: 4px;
}

.rs-verdict-note {
    font-family: var(--body);
    font-size: 14.5px;
    line-height: 1.5;
    margin-top: 10px;
    opacity: 0.9;
}

.rs-ring { flex: 0 0 auto; width: 132px; height: 132px; position: relative; }

.rs-ring svg { width: 132px; height: 132px; transform: rotate(-90deg); }

.rs-ring .track { fill: none; stroke-width: 12; stroke: rgba(255, 255, 255, 0.22); }
.rs-verdict.unknown .rs-ring .track { stroke: rgba(20, 24, 28, 0.15); }

.rs-ring .arc {
    fill: none;
    stroke-width: 12;
    stroke-linecap: round;
    stroke: var(--yellow);
    stroke-dasharray: 339.292;
    animation: rs-ring 1.1s cubic-bezier(0.2, 0.8, 0.2, 1) both;
}
.rs-verdict.unknown .rs-ring .arc { stroke: var(--asphalt); }

@keyframes rs-ring {
    from { stroke-dashoffset: 339.292; }
}

.rs-ring-label {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-family: var(--display);
    font-weight: 800;
    font-size: 34px;
    line-height: 1;
}

.rs-ring-label small {
    font-family: var(--body);
    font-weight: 500;
    font-size: 11px;
    opacity: 0.8;
    margin-top: 4px;
}


/* ---------- checks (two gates the sign must pass) ---------- */

.rs-card {
    background: var(--card);
    border: 1px solid var(--hairline);
    border-radius: 16px;
    padding: 22px 24px;
    margin-bottom: 18px;
    box-shadow: 0 4px 16px rgba(20, 24, 28, 0.05);
}

.rs-card-title {
    font-family: var(--display);
    font-weight: 700;
    font-size: 22px;
    color: var(--asphalt);
    margin-bottom: 14px;
}

.rs-gate { margin-bottom: 18px; }
.rs-gate:last-child { margin-bottom: 0; }

.rs-gate-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-family: var(--body);
    font-size: 14px;
    color: var(--asphalt);
    margin-bottom: 8px;
}

.rs-gate-head b { font-weight: 600; }

.rs-chip {
    font-family: var(--display);
    font-weight: 700;
    font-size: 15px;
    border-radius: 6px;
    padding: 2px 10px;
}
.rs-chip.pass { background: var(--green-light); color: var(--green); }
.rs-chip.fail { background: #fde7e7; color: var(--red); }

.rs-track {
    position: relative;
    height: 14px;
    border-radius: 7px;
    background: #e4e7e1;
}

.rs-fill {
    height: 100%;
    border-radius: 7px;
    animation: rs-grow 0.9s cubic-bezier(0.2, 0.8, 0.2, 1) both;
}
.rs-fill.pass { background: var(--green); }
.rs-fill.fail { background: var(--red); }

@keyframes rs-grow { from { width: 0; } }

.rs-tick {
    position: absolute;
    top: -5px;
    width: 3px;
    height: 24px;
    background: var(--asphalt);
    border-radius: 2px;
}

.rs-tick-note {
    font-family: var(--body);
    font-size: 12px;
    color: var(--asphalt-soft);
    margin-top: 6px;
}


/* ---------- top five ---------- */

.rs-rank {
    display: grid;
    grid-template-columns: 34px 1fr 64px;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
}
.rs-rank:last-child { margin-bottom: 0; }

.rs-rank-pos {
    font-family: var(--display);
    font-weight: 800;
    font-size: 20px;
    width: 34px;
    height: 34px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #e4e7e1;
    color: var(--asphalt);
}
.rs-rank.first .rs-rank-pos { background: var(--yellow); }

.rs-rank-name {
    font-family: var(--body);
    font-size: 14px;
    font-weight: 500;
    color: var(--asphalt);
    margin-bottom: 5px;
    line-height: 1.25;
}

.rs-rank-bar { height: 8px; border-radius: 4px; background: #e4e7e1; }
.rs-rank-bar > div {
    height: 100%;
    border-radius: 4px;
    background: var(--blue);
    animation: rs-grow 0.9s cubic-bezier(0.2, 0.8, 0.2, 1) both;
}
.rs-rank.first .rs-rank-bar > div { background: var(--green); }

.rs-rank-pct {
    font-family: var(--display);
    font-weight: 700;
    font-size: 18px;
    text-align: right;
    color: var(--asphalt);
}


/* ---------- empty state ---------- */

.rs-empty {
    background: var(--card);
    border: 2px dashed var(--hairline);
    border-radius: 18px;
    padding: 34px 30px;
}

.rs-empty-title {
    font-family: var(--display);
    font-weight: 700;
    font-size: 26px;
    color: var(--asphalt);
}

.rs-empty-text {
    font-family: var(--body);
    font-size: 14.5px;
    line-height: 1.6;
    color: var(--asphalt-soft);
    margin-top: 8px;
}

.rs-empty ol { margin: 14px 0 0 18px; padding: 0; color: var(--asphalt); font-size: 14.5px; line-height: 1.8; }


/* ---------- spec strip + footer ---------- */

.rs-specs {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-top: 6px;
}

.rs-spec {
    background: var(--asphalt);
    border-radius: 14px;
    padding: 18px 20px;
}

.rs-spec-value {
    font-family: var(--display);
    font-weight: 800;
    font-size: 30px;
    line-height: 1;
    color: var(--yellow);
}

.rs-spec-label {
    font-family: var(--body);
    font-size: 13px;
    color: #c9d0d6;
    margin-top: 6px;
}

.rs-footer {
    text-align: center;
    font-family: var(--body);
    font-size: 13px;
    color: var(--asphalt-soft);
    margin-top: 28px;
}


/* ---------- small screens ---------- */

@media (max-width: 900px) {
    .block-container { padding-top: 3.4rem !important; }
    .rs-sign { flex-direction: column; align-items: flex-start; padding: 26px 26px; gap: 14px; }
    .rs-sign-tags { margin-left: 0; flex-direction: row; flex-wrap: wrap; align-items: flex-start; }
    .rs-sign-title { font-size: 40px; }
    .rs-verdict { flex-direction: column; align-items: flex-start; }
    .rs-verdict-name { font-size: 32px; }
    .rs-specs { grid-template-columns: repeat(2, 1fr); }
}

@media (prefers-reduced-motion: reduce) {
    .rs-road, .rs-verdict, .rs-ring .arc, .rs-fill, .rs-rank-bar > div { animation: none; }
}
</style>
"""

st.html(STYLE)


# ============================================================
# MASTHEAD
# ============================================================

st.html(
    """
    <div class="rs-sign">
        <div class="rs-diamond"><span>!</span></div>
        <div>
            <div class="rs-sign-title">Roadsign Reader</div>
            <div class="rs-sign-sub">
                Upload a photo of a road sign. The model names it,
                or tells you plainly when it has never seen anything like it.
            </div>
        </div>
        <div class="rs-sign-tags">
            <div class="rs-tag">ResNet18</div>
            <div class="rs-tag">43 sign classes</div>
            <div class="rs-tag">Unknown-sign check</div>
        </div>
    </div>
    <div class="rs-road"></div>
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
        repo_type="model",
    )

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 43)

    state_dict = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    return model


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
        [0.229, 0.224, 0.225],
    ),
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
    started = time.perf_counter()

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_class = torch.max(probabilities, dim=1)
        top_probs, top_classes = torch.topk(probabilities, 5, dim=1)

        feature = extract_feature(model, image_tensor)

    predicted_class = predicted_class.item()
    confidence = confidence.item() * 100
    top_probs = top_probs[0].cpu().numpy() * 100
    top_classes = top_classes[0].cpu().numpy()

    similarity = 0.0
    closest_class = predicted_class

    if prototypes is not None:
        feature_np = feature[0].cpu().numpy()
        similarities = np.dot(prototypes, feature_np)
        closest_class = int(np.argmax(similarities))
        similarity = float(similarities[closest_class])

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
        "is_known": is_known,
        "elapsed_ms": (time.perf_counter() - started) * 1000,
    }


# ============================================================
# HTML BUILDERS
# ============================================================

RING_CIRCUMFERENCE = 339.292


def ring_html(percent):
    percent = max(0.0, min(percent, 100.0))
    offset = RING_CIRCUMFERENCE * (1 - percent / 100)

    return f"""
    <div class="rs-ring">
        <svg viewBox="0 0 120 120">
            <circle class="track" cx="60" cy="60" r="54"></circle>
            <circle class="arc" cx="60" cy="60" r="54"
                    style="stroke-dashoffset:{offset:.3f}"></circle>
        </svg>
        <div class="rs-ring-label">
            {percent:.0f}%
            <small>confidence</small>
        </div>
    </div>
    """


def verdict_html(result):
    if result["is_known"]:
        name = CLASS_NAMES[result["predicted_class"]]
        return f"""
        <div class="rs-verdict known">
            <div class="rs-verdict-body">
                <div class="rs-verdict-kicker">Sign recognised</div>
                <div class="rs-verdict-name">{name}</div>
                <div class="rs-verdict-note">
                    This sign matches one of the 43 GTSRB classes the model was trained on.
                </div>
            </div>
            {ring_html(result["confidence"])}
        </div>
        """

    closest = CLASS_NAMES[result["closest_class"]]
    return f"""
    <div class="rs-verdict unknown">
        <div class="rs-verdict-body">
            <div class="rs-verdict-kicker">Sign not recognised</div>
            <div class="rs-verdict-name">No confident match</div>
            <div class="rs-verdict-note">
                The closest class is <b>{closest}</b>, but the evidence is too weak to call it.
                Try a sharper, closer photo with the sign filling most of the frame.
            </div>
        </div>
        {ring_html(result["confidence"])}
    </div>
    """


def gate_html(title, shown, fraction, threshold_fraction, passed):
    fraction = max(0.0, min(fraction, 1.0))
    state = "pass" if passed else "fail"
    chip = "Pass" if passed else "Below limit"

    return f"""
    <div class="rs-gate">
        <div class="rs-gate-head">
            <span><b>{title}</b> &nbsp; {shown}</span>
            <span class="rs-chip {state}">{chip}</span>
        </div>
        <div class="rs-track">
            <div class="rs-fill {state}" style="width:{fraction * 100:.1f}%"></div>
            <div class="rs-tick" style="left:calc({threshold_fraction * 100:.1f}% - 1px)"></div>
        </div>
    </div>
    """


def checks_html(result):
    conf_ok = result["confidence"] >= CONFIDENCE_THRESHOLD
    sim_ok = result["similarity"] >= SIMILARITY_THRESHOLD

    conf_gate = gate_html(
        "Model confidence",
        f"{result['confidence']:.2f}%",
        result["confidence"] / 100,
        CONFIDENCE_THRESHOLD / 100,
        conf_ok,
    )

    sim_gate = gate_html(
        "Feature similarity",
        f"{result['similarity']:.4f}",
        result["similarity"],
        SIMILARITY_THRESHOLD,
        sim_ok,
    )

    return f"""
    <div class="rs-card">
        <div class="rs-card-title">Two checks decide the verdict</div>
        {conf_gate}
        {sim_gate}
        <div class="rs-tick-note">
            The black marker is the pass line:
            {CONFIDENCE_THRESHOLD:.0f}% confidence and {SIMILARITY_THRESHOLD:.2f} similarity.
            A sign must clear both.
        </div>
    </div>
    """


def top5_html(result):
    rows = []

    for i in range(5):
        class_id = int(result["top_classes"][i])
        probability = float(result["top_probs"][i])
        name = CLASS_NAMES[class_id]
        first = " first" if i == 0 else ""

        rows.append(f"""
        <div class="rs-rank{first}">
            <div class="rs-rank-pos">{i + 1}</div>
            <div>
                <div class="rs-rank-name">{name}</div>
                <div class="rs-rank-bar">
                    <div style="width:{min(probability, 100):.2f}%"></div>
                </div>
            </div>
            <div class="rs-rank-pct">{probability:.1f}%</div>
        </div>
        """)

    return f"""
    <div class="rs-card">
        <div class="rs-card-title">Five closest classes</div>
        {"".join(rows)}
    </div>
    """


# ============================================================
# MAIN WORKSPACE
# ============================================================

left_column, right_column = st.columns([0.92, 1.08], gap="large")


# ---------- left: upload + preview ----------

with left_column:
    st.html(
        """
        <div class="rs-h2">Your sign</div>
        <div class="rs-p">
            JPG or PNG. A close, well-lit shot works best.
        </div>
        """
    )

    uploaded_file = st.file_uploader(
        "Choose a traffic sign image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    image = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption=f"{uploaded_file.name}  ·  {image.width} × {image.height} px",
            use_container_width=True,
        )


# ---------- right: result ----------

with right_column:
    st.html(
        """
        <div class="rs-h2">What the model sees</div>
        <div class="rs-p">
            The result appears as soon as the image is uploaded.
        </div>
        """
    )

    if image is None:
        st.html(
            """
            <div class="rs-empty">
                <div class="rs-empty-title">Waiting for a sign</div>
                <div class="rs-empty-text">
                    Drop an image into the yellow box and the reading starts on its own.
                </div>
                <ol>
                    <li>The model scores the image against 43 known sign classes.</li>
                    <li>A second check compares the image with each class's average look.</li>
                    <li>Only a sign that passes both gets a name.</li>
                </ol>
            </div>
            """
        )

    else:
        try:
            with st.spinner("Reading the sign..."):
                model = load_model()
                prototypes = load_prototypes()
                result = predict(model, image, prototypes)

            st.html(verdict_html(result))
            st.html(checks_html(result))
            st.html(top5_html(result))

            st.caption(
                f"Read in {result['elapsed_ms']:.0f} ms on "
                f"{'GPU' if DEVICE.type == 'cuda' else 'CPU'}"
            )

        except Exception as error:
            st.error("The image could not be analysed. Try another file.")

            with st.expander("Technical details"):
                st.code(str(error))


# ============================================================
# SPEC STRIP + FOOTER
# ============================================================

st.html(
    f"""
    <div style="height:28px"></div>
    <div class="rs-h2">Under the hood</div>
    <div class="rs-p">What is running behind the page.</div>

    <div class="rs-specs">
        <div class="rs-spec">
            <div class="rs-spec-value">ResNet18</div>
            <div class="rs-spec-label">Classifier backbone</div>
        </div>
        <div class="rs-spec">
            <div class="rs-spec-value">GTSRB</div>
            <div class="rs-spec-label">Training dataset</div>
        </div>
        <div class="rs-spec">
            <div class="rs-spec-value">43</div>
            <div class="rs-spec-label">Sign classes</div>
        </div>
        <div class="rs-spec">
            <div class="rs-spec-value">{'GPU' if DEVICE.type == 'cuda' else 'CPU'}</div>
            <div class="rs-spec-label">Running on</div>
        </div>
    </div>

    <div class="rs-footer">Roadsign Reader · built with PyTorch and Streamlit</div>
    """
)
