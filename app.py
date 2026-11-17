import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision import transforms
from huggingface_hub import hf_hub_download
from PIL import Image


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Traffic Sign Recognizer",
    page_icon="🚦",
    layout="centered"
)


# =========================
# CONFIGURATION
# =========================

MODEL_REPO = "tanim4hmed/traffic-sign-recognizer-v2"
MODEL_FILE = "traffic_sign_resnet18_final.pth"

CONFIDENCE_THRESHOLD = 70.0


# =========================
# GTSRB CLASS NAMES
# =========================

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


# =========================
# IMAGE TRANSFORMATION
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        repo_type="model"
    )

    model = resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        43
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu"
        )
    )

    model.eval()

    return model


# =========================
# PREDICTION
# =========================

def predict(image, model):

    image_tensor = transform(image).unsqueeze(0)

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

    top5_prob, top5_classes = torch.topk(
        probabilities,
        5,
        dim=1
    )

    return (
        predicted_class.item(),
        confidence.item() * 100,
        top5_classes[0].tolist(),
        top5_prob[0].tolist()
    )


# =========================
# USER INTERFACE
# =========================

st.title("🚦 Traffic Sign Recognizer")

st.write(
    "Upload a traffic sign image and the AI model "
    "will identify the sign."
)

st.info(
    "Model: ResNet18 | "
    "GTSRB Dataset | "
    "43 Traffic Sign Classes"
)

uploaded_file = st.file_uploader(
    "📷 Upload a traffic sign image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Recognize Traffic Sign"):

        with st.spinner("Analyzing image..."):

            model = load_model()

            predicted_class, confidence, top5_classes, top5_probs = predict(
                image,
                model
            )

        st.divider()

        st.subheader("Prediction")

        st.success(
            f"🚦 {CLASS_NAMES[predicted_class]}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        if confidence >= CONFIDENCE_THRESHOLD:

            st.success(
                "✅ Traffic sign recognized successfully!"
            )

        else:

            st.warning(
                "⚠️ Model is not confident enough. "
                "Try another image."
            )

        st.subheader("Top 5 Predictions")

        for class_id, probability in zip(
            top5_classes,
            top5_probs
        ):

            percentage = probability * 100

            st.write(
                f"**{CLASS_NAMES[class_id]}** — "
                f"{percentage:.2f}%"
            )

            st.progress(
                min(int(percentage), 100)
            )


st.divider()

st.caption(
    "Traffic Sign Recognition System | "
    "ResNet18 trained on GTSRB"
)
