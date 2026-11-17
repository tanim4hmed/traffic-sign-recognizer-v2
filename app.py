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

st.set_page_config(
    page_title="Traffic Sign Recognizer V2",
    page_icon="🚦",
    layout="centered"
)

st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #777;
    margin-bottom: 30px;
}

.result-known {
    padding: 20px;
    border-radius: 15px;
    border: 2px solid #28a745;
    background-color: #f0fff4;
    margin-top: 20px;
}

.result-unknown {
    padding: 20px;
    border-radius: 15px;
    border: 2px solid #dc3545;
    background-color: #fff5f5;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🚦 Traffic Sign Recognizer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered recognition using ResNet18 • 43 GTSRB classes</div>',
    unsafe_allow_html=True
)

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

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="tanim4hmed/traffic-sign-recognizer-v2",
        filename="traffic_sign_resnet18_final.pth",
        repo_type="model"
    )

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 43)

    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model

@st.cache_data
def load_prototypes():
    path = "traffic_sign_prototypes.npz"

    if not os.path.exists(path):
        return None

    data = np.load(path)
    return data["prototypes"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

def extract_feature(model, image):
    image_tensor = transform(image).unsqueeze(0).to(device)

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
        feature = torch.flatten(x, 1)
        feature = F.normalize(feature, p=2, dim=1)

    return feature.cpu().numpy()[0]

def predict(model, image, prototypes):
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)

    confidence, predicted_class = torch.max(
        probabilities, dim=1
    )

    confidence = confidence.item() * 100
    predicted_class = predicted_class.item()

    top5_prob, top5_classes = torch.topk(
        probabilities, 5, dim=1
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

    feature = extract_feature(model, image)

    similarities = np.dot(prototypes, feature)

    closest_class = int(np.argmax(similarities))
    similarity = float(similarities[closest_class])

    is_known = (
        confidence >= CONFIDENCE_THRESHOLD
        and similarity >= SIMILARITY_THRESHOLD
    )

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "closest_class": closest_class,
        "similarity": similarity,
        "is_known": is_known,
        "top5": top5
    }

with st.spinner("Loading AI model..."):
    model = load_model()

prototypes = load_prototypes()

st.markdown("### 📤 Upload a Traffic Sign")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear traffic sign image."
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Traffic Sign",
        use_container_width=True
    )

    st.markdown("---")

    if prototypes is None:

        st.error(
            "Prototype file is missing. "
            "Please add traffic_sign_prototypes.npz "
            "to the GitHub repository."
        )

    else:

        if st.button(
            "🔍 Recognize Traffic Sign",
            use_container_width=True
        ):

            with st.spinner("Analyzing image..."):
                result = predict(
                    model,
                    image,
                    prototypes
                )

            if not result["is_known"]:

                st.markdown(
                    """
                    <div class="result-unknown">
                        <h2>⚠️ Unknown Sign</h2>
                        <p>
                        This sign is not confidently recognized
                        as one of the 43 GTSRB classes.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Model Confidence",
                        f"{result['confidence']:.2f}%"
                    )

                with col2:
                    st.metric(
                        "Feature Similarity",
                        f"{result['similarity']:.4f}"
                    )

                st.info(
                    "The model's highest prediction was "
                    f"**{CLASS_NAMES[result['predicted_class']]}**, "
                    "but the sign did not sufficiently match "
                    "the known GTSRB classes."
                )

            else:

                predicted_name = CLASS_NAMES[
                    result["predicted_class"]
                ]

                st.markdown(
                    f"""
                    <div class="result-known">
                        <h2>✅ Traffic Sign Recognized</h2>
                        <h3>{predicted_name}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Confidence",
                        f"{result['confidence']:.2f}%"
                    )

                with col2:
                    st.metric(
                        "Feature Similarity",
                        f"{result['similarity']:.4f}"
                    )

            st.markdown("### 📊 Top 5 Predictions")

            for item in result["top5"]:

                st.write(
                    f"**{item['name']}** — "
                    f"{item['confidence']:.2f}%"
                )

                st.progress(
                    min(int(item["confidence"]), 100)
                )

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#888;">
        <p><b>Traffic Sign Recognizer V2</b></p>
        <p>ResNet18 • GTSRB • 43 Classes</p>
    </div>
    """,
    unsafe_allow_html=True
)
