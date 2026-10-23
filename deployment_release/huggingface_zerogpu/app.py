"""BirdLens: a clean Gradio UI for the selected COS30082 Model 8 checkpoint.

Run locally from the project root:
    .\\.venv\\Scripts\\python.exe app\\app.py

For Hugging Face ZeroGPU, use scripts/build_free_deployments.py to assemble the Space.
"""
from __future__ import annotations

import html

# Hugging Face ZeroGPU provides this package; import before torch/torchvision.
import spaces
import gradio as gr
from PIL import Image

from inference import BirdPredictor


# Loading once on startup avoids silently falling back to untrained weights.
# On ZeroGPU, importing spaces enables module-level CUDA placement without
# permanently occupying a physical GPU. GPU is allocated for @spaces.GPU calls.
# The preselected checkpoint is loaded exactly once at startup.
PREDICTOR = BirdPredictor(device="cuda")


CSS = """
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; }
#hero { padding: 24px 28px; border-radius: 20px; margin-bottom: 14px;
    background: linear-gradient(125deg, #123b35 0%, #18594b 65%, #216a80 100%);
    color: #f3fff9 !important; box-shadow: 0 10px 32px rgba(12,49,40,.13); }
#hero h1 { color: #f3fff9 !important; font-size: 2.1rem; margin: 9px 0; }
#hero p { color: #e3f4ee !important; margin-bottom: 4px; }
#hero .eyebrow { font-size: .8rem; text-transform: uppercase; letter-spacing: .16em;
    color: #b9e7d3; font-weight: 750; }
#facts { padding: 8px 2px; color: var(--body-text-color); font-size: .94rem; }
#footnote { font-size: .88rem; opacity: .9; padding: 6px 4px; }
"""


@spaces.GPU(duration=20)
def classify(image: Image.Image | None):
    if image is None:
        return {}, "**Upload a bird image** to see the predicted species and five most likely classes."
    try:
        top5 = PREDICTOR.predict_top5(image)
    except (ValueError, TypeError) as exc:
        raise gr.Error(str(exc)) from exc
    # Gradio Label displays five bars; probabilities are from the 200-way softmax.
    confidence_map = dict(top5)
    best_species, confidence = top5[0]
    summary = (
        f"### Predicted species: {html.escape(best_species)}\n\n"
        f"**Model score:** {confidence * 100:.1f}%\n\n"
        "The percentages are model probability scores, **not guaranteed correctness**. "
        "The classifier covers only the 200 bird species in this assignment dataset."
    )
    return confidence_map, summary


def build_demo():
    with gr.Blocks(title="BirdLens | 200 Bird Species") as demo:
        gr.HTML("""
<div id="hero">
    <div class="eyebrow">COS30082 · Applied Machine Learning</div>
    <h1>🐦 BirdLens</h1>
    <p>Identify birds from an image using a pretrained ResNet18 classifier with Mixup training.</p>
</div>
""")
        with gr.Row(equal_height=False):
            with gr.Column(scale=6, min_width=300):
                photo = gr.Image(
                    label="Bird photograph", type="pil", image_mode="RGB",
                    sources=["upload"], height=340,
                    placeholder="Upload a clear photo containing a bird",
                )
                with gr.Row():
                    identify = gr.Button("Identify bird", variant="primary", scale=3)
                    clear = gr.ClearButton(value="Clear", components=[], scale=1)
            with gr.Column(scale=5, min_width=300):
                scores = gr.Label(label="Top 5 predicted species", num_top_classes=5)
                summary = gr.Markdown(
                    "**Upload a bird image** to see the predicted species and five most likely classes."
                )
                gr.Markdown(
                    "**Tip:** A clear, well-lit photo with the bird visible usually gives the "
                    "model more useful detail to work with."
                )

        clear.add([photo, scores, summary])
        identify.click(classify, inputs=photo, outputs=[scores, summary],
                       api_name="predict", concurrency_limit=1)
        with gr.Row(elem_id="facts"):
            gr.Markdown(
                "**200 species** · **ResNet18 + Mixup** · **224 × 224 inference** "
                "· **Model 8 / epoch 18**"
            )
        gr.Markdown(
            "**Evaluation:** 58.55% Top-1 accuracy on 1,204 held-out test images. "
            "**Limitations:** Results can be wrong, including when the score is high. "
            "Photos of unsupported species will still receive a prediction from the known classes. "
            "This demo is an educational classifier, not a verified wildlife identification service.",
            elem_id="footnote",
        )
    return demo


demo = build_demo()

if __name__ == "__main__":
    # The deployed Space serves this Gradio app publicly; no separate tunnel.
    demo.launch(server_name="0.0.0.0", share=False, max_file_size="10mb", show_error=True)
