"""Reward text as an opaque GPU image, using the existing camera overlay's XR path."""

import textwrap
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .xr_control_status import QuestControlStatus


@lru_cache(maxsize=1)
def _font():
    try:
        return ImageFont.truetype("DejaVuSansMono.ttf", 20)
    except OSError:
        return ImageFont.load_default(size=20)


def reward_image(text, width=768, height=864):
    """Rasterize all rows, including zeros, without creating an Isaac/Kit context."""
    font = _font()
    columns = max(1, int((width - 32) / font.getlength("M")))
    lines = [part for line in text.splitlines()
             for part in (textwrap.wrap(line, columns, replace_whitespace=False) or [""])]
    line_height = 26
    # Fit custom reward lists too, rather than silently cropping the lower rows.
    content = Image.new("RGBA", (width, max(height, 32 + len(lines) * line_height)), (16, 22, 32, 255))
    draw = ImageDraw.Draw(content)
    draw.rectangle((1, 1, width - 2, content.height - 2), outline=(60, 220, 240, 255), width=3)
    for index, line in enumerate(lines):
        color = (255, 255, 255, 255)
        if index == 0 or line.startswith("TOTAL:"):
            color = (90, 235, 255, 255)
        elif ": -" in line:
            color = (255, 155, 120, 255)
        draw.text((16, 16 + index * line_height), line, font=font, fill=color)
    if content.height != height:
        content = content.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(content).copy()


class QuestRewardPanel(QuestControlStatus):
    def __init__(self, *, forward_axis="-z"):
        import omni.ui as ui
        # Same provider and image widget as the functioning wrist camera panels.
        self._provider = ui.ByteImageProvider()
        self._gpu_frames = []
        self._last_image_text = None
        self.upload_count = 0
        width, height = 768, 864
        provider = self._provider

        class RewardWidget(ui.Frame):
            def __init__(self):
                super().__init__(width=width, height=height)
                with self:
                    ui.ImageWithProvider(provider, fill_policy=ui.IwpFillPolicy.IWP_STRETCH,
                                         width=width, height=height)

        # Share the camera-overlay layer and tracked head path, including when
        # camera sensors themselves are disabled. This adds only one panel.
        super().__init__(layer_name="kuavo_camera_overlay", width=width, height=height,
                         pixels_per_cm=40., position=(0., -4., -28. if forward_axis == "-z" else 28.),
                         widget_type=RewardWidget)
        self.update("RL REWARD | READY\nA: start/pause\nY: show/hide reward panel")

    def update(self, text, *, alert=False):
        import torch
        if text != self._last_image_text:
            rgba = reward_image(text)
            frame = torch.as_tensor(rgba, device="cuda:0").contiguous()
            self._provider.set_bytes_data_from_gpu(frame.data_ptr(), [rgba.shape[1], rgba.shape[0]])
            # Keep the previous upload alive until the next rendered frame too.
            self._gpu_frames = (self._gpu_frames + [frame])[-2:]
            self._last_image_text = text
            self.upload_count += 1
            if self.component.scene_widget is not None:
                self.component.scene_widget.invalidate()
        return self.component.widget is not None

    def describe(self):
        print(f"[XR REWARD TEXTURE] uploads={self.upload_count} "
              f"layer={self.layer.is_visible()} panel={self.container.visible} "
              f"widget_ready={self.component.widget is not None} "
              f"system={self.container.scene_view.system_path}", flush=True)
