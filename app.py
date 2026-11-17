import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np
import os

# =========================================================

# PAGE CONFIG

# =========================================================

st.set_page_config(
page_title="TrafficVision AI",
page_icon="🚦",
layout="wide",
initial_sidebar_state="collapsed"
)

# =========================================================

# MODERN UI STYLE

# =========================================================

st.markdown("""

\<style>

.stApp {
&#x20;   background:
&#x20;       radial-gradient(circle at 15% 10%, rgba(59,130,246,0.12), transparent 30%),
&#x20;       radial-gradient(circle at 85% 20%, rgba(139,92,246,0.10), transparent 30%),
&#x20;       \#080b12;
&#x20;   color: #f5f7fb;
}

.block-container {
&#x20;   max-width: 1150px;
&#x20;   padding-top: 2rem;
&#x20;   padding-bottom: 3rem;
}

/\* Header \*/

.hero {
&#x20;   text-align: center;
&#x20;   padding: 20px 10px 35px 10px;
}

.hero-title {
&#x20;   font-size: 48px;
&#x20;   font-weight: 800;
&#x20;   letter-spacing: -1px;
&#x20;   margin-bottom: 8px;
}

.hero-title span {
&#x20;   color: #60a5fa;
}

.hero-subtitle {
&#x20;   color: #9ca3af;
&#x20;   font-size: 18px;
}

.status {
&#x20;   display: inline-block;
&#x20;   margin-top: 18px;
&#x20;   padding: 7px 15px;
&#x20;   border-radius: 999px;
&#x20;   background: rgba(34,197,94,0.12);
&#x20;   border: 1px solid rgba(34,197,94,0.3);
&#x20;   color: #4ade80;
&#x20;   font-size: 13px;
&#x20;   font-weight: 700;
}

/\* Cards \*/

.glass-card {
&#x20;   background: rgba(17,24,39,0.72);
&#x20;   border: 1px solid rgba(255,255,255,0.08);
&#x20;   border-radius: 22px;
&#x20;   padding: 25px;
&#x20;   box-shadow: 0 15px 45px rgba(0,0,0,0.25);
&#x20;   backdrop-filter: blur(12px);
}

.section-title {
&#x20;   font-size: 21px;
&#x20;   font-weight: 700;
&#x20;   margin-bottom: 15px;
}

/\* Upload \*/

.upload-box {
&#x20;   text-align: center;
&#x20;   padding: 35px 20px 20px 20px;
}

.upload-icon {
&#x20;   font-size: 45px;
&#x20;   margin-bottom: 10px;
}

.upload-title {
&#x20;   font-size: 25px;
&#x20;   font-weight: 700;
}

.upload-text {
&#x20;   color: #9ca3af;
&#x20;   margin-bottom: 15px;
}

/\* Result \*/

.result-known {
&#x20;   background: rgba(34,197,94,0.08);
&#x20;   border: 1px solid rgba(34,197,94,0.35);
&#x20;   border-radius: 18px;
&#x20;   padding: 22px;
}

.result-unknown {
&#x20;   background: rgba(239,68,68,0.08);
&#x20;   border: 1px solid rgba(239,68,68,0.35);
&#x20;   border-radius: 18px;
&#x20;   padding: 22px;
}

.result-status {
&#x20;   font-size: 14px;
&#x20;   font-weight: 800;
&#x20;   letter-spacing: 1px;
}

.result-name {
&#x20;   font-size: 28px;
&#x20;   font-weight: 800;
&#x20;   margin-top: 8px;
}

/\* Metrics \*/

.metric-card {
&#x20;   background: rgba(255,255,255,0.035);
&#x20;   border: 1px solid rgba(255,255,255,0.07);
&#x20;   border-radius: 15px;
&#x20;   padding: 17px;
&#x20;   text-align: center;
}

.metric-label {
&#x20;   color: #9ca3af;
&#x20;   font-size: 13px;
}

.metric-value {
&#x20;   font-size: 24px;
&#x20;   font-weight: 800;
&#x20;   margin-top: 5px;
}

/\* Top predictions \*/

.prediction-row {
&#x20;   margin: 14px 0;
}

.prediction-header {
&#x20;   display: flex;
&#x20;   justify-content: space-between;
&#x20;   margin-bottom: 6px;
&#x20;   font-size: 14px;
}

.prediction-bar {
&#x20;   height: 9px;
&#x20;   border-radius: 10px;
&#x20;   background: rgba(255,255,255,0.08);
&#x20;   overflow: hidden;
}

.prediction-fill {
&#x20;   height: 100%;
&#x20;   border-radius: 10px;
&#x20;   background: linear-gradient(90deg, #3b82f6, #8b5cf6);
}

/\* How it works \*/

.step-card {
&#x20;   text-align: center;
&#x20;   background: rgba(255,255,255,0.035);
&#x20;   border: 1px solid rgba(255,255,255,0.07);
&#x20;   border-radius: 16px;
&#x20;   padding: 18px 10px;
}

.step-icon {
&#x20;   font-size: 28px;
}

.step-title {
&#x20;   font-weight: 700;
&#x20;   margin-top: 8px;
}

.step-text {
&#x20;   color: #9ca3af;
&#x20;   font-size: 13px;
}

/\* Footer \*/

.footer {
&#x20;   text-align: center;
&#x20;   color: #6b7280;
&#x20;   margin-top: 45px;
&#x20;   padding-top: 20px;
&#x20;   border-top: 1px solid rgba(255,255,255,0.06);
}

/\* Button \*/

.stButton > button {
&#x20;   width: 100%;
&#x20;   border-radius: 12px;
&#x20;   height: 48px;
&#x20;   font-weight: 700;
&#x20;   border: 1px solid rgba(96,165,250,0.35);
&#x20;   background: linear-gradient(90deg, #2563eb, #7c3aed);
&#x20;   color: white;
}

.stButton > button\:hover {
&#x20;   border-color: #60a5fa;
&#x20;   color: white;
}

/\* Mobile \*/

@media (max-width: 700px) {

&#x20;   .hero-title {
&#x20;       font-size: 36px;
&#x20;   }

&#x20;   .hero-subtitle {
&#x20;       font-size: 15px;
&#x20;   }

&#x20;   .result-name {
&#x20;       font-size: 23px;
&#x20;   }

}

\</style>

""", unsafe_allow_html=True)

# =========================================================

# HEADER

# =========================================================

st.markdown("""

\<div class="hero">

```
<div class="hero-title">
    🚦 <span>TrafficVision AI</span>
</div>

<div class="hero-subtitle">
    Intelligent Traffic Sign Recognition & Unknown Sign Detection
</div>

<div class="status">
    ● AI MODEL ONLINE
</div>
```

\</div>
""", unsafe_allow_html=True)

# =========================================================

# CLASS NAMES

# =========================================================

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
"Right-of-way at next intersection",
"Priority road",
"Yield",
"Stop",
"No vehicles",
"Vehicles over 3.5 metric tons prohibited",
"No entry",
"General caution",
"Dangerous curve left",
"Dangerous curve right",
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

CONFIDENCE_THRESHOLD = 70.0
SIMILARITY_THRESHOLD = 0.90

# =========================================================

# DEVICE

# =========================================================

device = torch.device(
"cuda" if torch.cuda.is_available() else "cpu"
)

# =========================================================

# LOAD MODEL

# =========================================================

@st.cache_resource
def load_model():

```
model_path = hf_hub_download(
    repo_id="tanim4hmed/traffic-sign-recognizer-v2",
    filename="traffic_sign_resnet18_final.pth",
    repo_type="model"
)

model = resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    43
)

state_dict = torch.load(
    model_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)

model = model.to(device)
model.eval()

return model
```

# =========================================================

# LOAD PROTOTYPES

# =========================================================

@st.cache_data
def load_prototypes():

```
path = "traffic_sign_prototypes.npz"

if not os.path.exists(path):
    return None

data = np.load(path)

return data["prototypes"]
```

# =========================================================

# IMAGE TRANSFORMATION

# =========================================================

transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize(
[0.485, 0.456, 0.406],
[0.229, 0.224, 0.225]
)
])

# =========================================================

# FEATURE EXTRACTION

# =========================================================

def extract_feature(model, image):

```
image_tensor = (
    transform(image)
    .unsqueeze(0)
    .to(device)
)

with torch.no_grad():

    x = model.conv1(image_tensor)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)

    feature = torch.flatten(
        x,
        1
    )

    feature = F.normalize(
        feature,
        p=2,
        dim=1
    )

return feature.cpu().numpy()[0]
```

# =========================================================

# PREDICTION

# =========================================================

def predict(model, image, prototypes):

```
image_tensor = (
    transform(image)
    .unsqueeze(0)
    .to(device)
)

with torch.no_grad():

    outputs = model(image_tensor)

    probabilities = F.softmax(
        outputs,
        dim=1
    )

confidence, predicted_class = torch.max(
    probabilities,
    dim=1
)

confidence = confidence.item() * 100

predicted_class = predicted_class.item()

top5_prob, top5_classes = torch.topk(
    probabilities,
    5,
    dim=1
)

top5 = []

for prob, class_id in zip(
    top5_prob[0].cpu().tolist(),
    top5_classes[0].cpu().tolist()
):

    top5.append({
        "name": CLASS_NAMES[class_id],
        "confidence": prob * 100
    })

feature = extract_feature(
    model,
    image
)

similarities = np.dot(
    prototypes,
    feature
)

closest_class = int(
    np.argmax(similarities)
)

similarity = float(
    similarities[closest_class]
)

is_known = (
    confidence >= CONFIDENCE_THRESHOLD
    and
    similarity >= SIMILARITY_THRESHOLD
)

return {
    "predicted_class": predicted_class,
    "confidence": confidence,
    "closest_class": closest_class,
    "similarity": similarity,
    "is_known": is_known,
    "top5": top5
}
```

# =========================================================

# LOAD AI SYSTEM

# =========================================================

with st.spinner("Initializing TrafficVision AI..."):

```
model = load_model()

prototypes = load_prototypes()
```

# =========================================================

# UPLOAD SECTION

# =========================================================

st.markdown("""

\<div class="glass-card upload-box">

```
<div class="upload-icon">📤</div>

<div class="upload-title">
    Upload a Traffic Sign
</div>

<div class="upload-text">
    Upload a clear JPG, JPEG or PNG image
    for AI-powered recognition.
</div>
```

\</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
"Choose an image",
type=["jpg", "jpeg", "png"],
label_visibility="collapsed"
)

# =========================================================

# PROCESS IMAGE

# =========================================================

if uploaded_file:

```
image = Image.open(
    uploaded_file
).convert("RGB")

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns(
    [1, 1],
    gap="large"
)

# -----------------------------------------------------
# IMAGE
# -----------------------------------------------------

with left:

    st.markdown(
        '<div class="section-title">🖼️ Uploaded Image</div>',
        unsafe_allow_html=True
    )

    st.image(
        image,
        use_container_width=True
    )

# -----------------------------------------------------
# ANALYSIS
# -----------------------------------------------------

with right:

    st.markdown(
        '<div class="section-title">🧠 AI Analysis</div>',
        unsafe_allow_html=True
    )

    if prototypes is None:

        st.error(
            "Prototype file is missing. "
            "Please add traffic_sign_prototypes.npz "
            "to the GitHub repository."
        )

    else:

        if st.button(
            "🔍 Analyze Traffic Sign"
        ):

            with st.spinner(
                "Analyzing traffic sign..."
            ):

                result = predict(
                    model,
                    image,
                    prototypes
                )

            # =================================================
            # UNKNOWN
            # =================================================

            if not result["is_known"]:

                st.markdown(f"""
                <div class="result-unknown">

                    <div class="result-status">
                        ⚠ UNKNOWN SIGN
                    </div>

                    <div class="result-name">
                        Outside GTSRB Classes
                    </div>

                    <p style="color:#9ca3af;">
                        The image does not sufficiently
                        match the recognized 43 traffic
                        sign classes.
                    </p>

                </div>
                """, unsafe_allow_html=True)

            # =================================================
            # KNOWN
            # =================================================

            else:

                predicted_name = CLASS_NAMES[
                    result["predicted_class"]
                ]

                st.markdown(f"""
                <div class="result-known">

                    <div class="result-status">
                        ✓ RECOGNIZED
                    </div>

                    <div class="result-name">
                        {predicted_name}
                    </div>

                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # =================================================
            # METRICS
            # =================================================

            metric1, metric2 = st.columns(2)

            with metric1:

                st.markdown(f"""
                <div class="metric-card">

                    <div class="metric-label">
                        Model Confidence
                    </div>

                    <div class="metric-value">
                        {result["confidence"]:.2f}%
                    </div>

                </div>
                """, unsafe_allow_html=True)

            with metric2:

                st.markdown(f"""
                <div class="metric-card">

                    <div class="metric-label">
                        Feature Similarity
                    </div>

                    <div class="metric-value">
                        {result["similarity"]:.4f}
                    </div>

                </div>
                """, unsafe_allow_html=True)

# =========================================================
# TOP 5
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">📊 Top 5 Predictions</div>',
    unsafe_allow_html=True
)

if prototypes is not None and "result" in locals():

    for item in result["top5"]:

        percentage = item["confidence"]

        st.markdown(f"""
        <div class="prediction-row">

            <div class="prediction-header">

                <span>
                    {item["name"]}
                </span>

                <span>
                    <b>{percentage:.2f}%</b>
                </span>

            </div>

            <div class="prediction-bar">

                <div
                    class="prediction-fill"
                    style="width:{min(percentage, 100)}%"
                ></div>

            </div>

        </div>
        """, unsafe_allow_html=True)
```

# =========================================================

# HOW IT WORKS

# =========================================================

st.markdown("\<br>\<br>", unsafe_allow_html=True)

st.markdown(
'\<div class="section-title">⚙️ How It Works\</div>',
unsafe_allow_html=True
)

step1, step2, step3, step4 = st.columns(4)

steps = [
(
step1,
"📤",
"Upload",
"Upload a traffic sign image."
),
(
step2,
"🧠",
"ResNet18",
"AI model predicts the sign."
),
(
step3,
"📐",
"Similarity",
"Visual features are compared."
),
(
step4,
"✅",
"Decision",
"Known or unknown sign."
)
]

for column, icon, title, description in steps:

```
with column:

    st.markdown(f"""
    <div class="step-card">

        <div class="step-icon">
            {icon}
        </div>

        <div class="step-title">
            {title}
        </div>

        <div class="step-text">
            {description}
        </div>

    </div>
    """, unsafe_allow_html=True)
```

# =========================================================

# MODEL INFO

# =========================================================

st.markdown("\<br>\<br>", unsafe_allow_html=True)

info1, info2, info3, info4 = st.columns(4)

model_info = [
(info1, "MODEL", "ResNet18"),
(info2, "DATASET", "GTSRB"),
(info3, "CLASSES", "43"),
(info4, "UNKNOWN AI", "Enabled")
]

for column, label, value in model_info:

```
with column:

    st.markdown(f"""
    <div class="metric-card">

        <div class="metric-label">
            {label}
        </div>

        <div class="metric-value">
            {value}
        </div>

    </div>
    """, unsafe_allow_html=True)
```

# =========================================================

# FOOTER

# =========================================================

st.markdown("""

\<div class="footer">

```
<b>TrafficVision AI</b>
<br>
ResNet18 • GTSRB • 43 Classes • Unknown Sign Detection
<br><br>
Built with PyTorch & Streamlit
```

\</div> 
