"""Reward text as an opaque GPU image, using the existing camera overlay's XR path."""

import textwrap
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .xr_control_status import QuestControlStatus


@lru_cache(maxsize=4)
def _font(size=20):
    try:
        return ImageFont.truetype("DejaVuSansMono.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def reward_image(text, width=768, height=864, *, headline=None, checks=(), failure=False):
    """Rasterize all rows, including zeros, without creating an Isaac/Kit context."""
    if headline is not None:
        # Reserve a separate header: long diagnostic/reward lists must NEVER
        # shrink the failure reason or success checklist alongside the body.
        heading_font, check_font = _font(36), _font(24)
        heading_lines = textwrap.wrap(headline, max(1, int((width - 32) / heading_font.getlength("M"))))
        check_lines = [part for line in checks for part in
                       (textwrap.wrap(line, max(1, int((width - 32) / check_font.getlength("M")))) or [""])]
        header_height = 28 + len(heading_lines) * 46 + len(check_lines) * 31
        header_height = min(header_height, height - 100)
        image = Image.new("RGBA", (width, height), (16, 22, 32, 255))
        draw = ImageDraw.Draw(image)
        y = 12
        for line in heading_lines:
            color = (255, 90, 90, 255) if failure else ((100, 255, 140, 255) if headline == "SUCCESS" else (90, 235, 255, 255))
            draw.text((16, y), line, font=heading_font, fill=color)
            y += 46
        for line in check_lines:
            if y + 31 > header_height:
                break
            color = ((255, 195, 80, 255) if line.startswith("NEED:") else
                     (120, 255, 160, 255) if line.startswith("OK:") else (255, 220, 220, 255))
            draw.text((16, y), line, font=check_font, fill=color)
            y += 31
        image.paste(Image.fromarray(reward_image(text, width, height - header_height)), (0, header_height))
        return np.asarray(image).copy()
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
                         pixels_per_cm=40., position=(-5., -4., -28. if forward_axis == "-z" else 28.),
                         widget_type=RewardWidget)
        self.update("RL REWARD | READY\nA: start/pause\nY: show/hide reward panel")

    def update(self, text, *, alert=False, headline=None, checks=(), failure=False):
        import torch
        key = (text, headline, tuple(checks), failure)
        if key != self._last_image_text:
            rgba = reward_image(text, headline=headline, checks=checks, failure=failure)
            frame = torch.as_tensor(rgba, device="cuda:0").contiguous()
            self._provider.set_bytes_data_from_gpu(frame.data_ptr(), [rgba.shape[1], rgba.shape[0]])
            # Keep the previous upload alive until the next rendered frame too.
            self._gpu_frames = (self._gpu_frames + [frame])[-2:]
            self._last_image_text = key
            self.upload_count += 1
            if self.component.scene_widget is not None:
                self.component.scene_widget.invalidate()
        return self.component.widget is not None

    def describe(self):
        print(f"[XR REWARD TEXTURE] uploads={self.upload_count} "
              f"layer={self.layer.is_visible()} panel={self.container.visible} "
              f"widget_ready={self.component.widget is not None} "
              f"system={self.container.scene_view.system_path}", flush=True)
