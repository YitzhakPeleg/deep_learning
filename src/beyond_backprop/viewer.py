import os
from pathlib import Path

import gradio as gr
import numpy as np

from beyond_backprop.mnist import Split, load_mnist

_DATA_DIR: Path = Path(os.environ.get("DATA_DIR", "data"))

_DATA: dict[Split, tuple[np.ndarray, np.ndarray]] = {
    Split.TRAIN: load_mnist(_DATA_DIR, Split.TRAIN),
    Split.TEST:  load_mnist(_DATA_DIR, Split.TEST),
}

_KEYBOARD_JS: str = (Path(__file__).parent / "viewer.js").read_text()


def _get(
    split: str,
    idx: int | float,
) -> tuple[np.ndarray, str, gr.update]:
    """Return the image, caption, and updated slider for a given split and index.

    Parameters
    ----------
    split : str
        Dataset split; coerced to ``Split``.
    idx : int | float
        Requested index; clamped to valid range.

    Returns
    -------
    image : np.ndarray
        Shape ``(28, 28)``, dtype ``uint8``.
    caption : str
        Human-readable label and index info.
    slider_update : gr.update
        Updated slider ``maximum`` and ``value``.
    """
    images, labels = _DATA[Split(split)]
    idx = int(np.clip(int(idx), 0, len(images) - 1))
    caption = f"Label: {int(labels[idx])}  ·  Index: {idx} / {len(images) - 1}"
    return images[idx], caption, gr.update(maximum=len(images) - 1, value=idx)


def main() -> None:
    """Launch the MNIST image viewer on http://localhost:7860.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    with gr.Blocks(title="MNIST Viewer") as app:
        split_radio = gr.Radio(
            choices=[Split.TRAIN, Split.TEST],
            value=Split.TRAIN,
            label="Split",
        )
        idx_slider = gr.Slider(minimum=0, maximum=59_999, step=1, value=0, label="Index")
        with gr.Row():
            prev_btn = gr.Button("← Prev", elem_id="prev-btn")
            next_btn = gr.Button("Next →", elem_id="next-btn")
        image_out = gr.Image(label="Image", width=280, height=280)
        caption_out = gr.Textbox(label="Info", interactive=False)

        outputs = [image_out, caption_out, idx_slider]

        split_radio.change(_get, inputs=[split_radio, idx_slider], outputs=outputs)
        idx_slider.release(_get, inputs=[split_radio, idx_slider], outputs=outputs)
        prev_btn.click(
            lambda s, i: _get(s, int(i) - 1),
            inputs=[split_radio, idx_slider],
            outputs=outputs,
        )
        next_btn.click(
            lambda s, i: _get(s, int(i) + 1),
            inputs=[split_radio, idx_slider],
            outputs=outputs,
        )

        # Populate on load
        app.load(_get, inputs=[split_radio, idx_slider], outputs=outputs)

    app.launch(js=_KEYBOARD_JS)


if __name__ == "__main__":
    main()
