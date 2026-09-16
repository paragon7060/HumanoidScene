import importlib.util
from pathlib import Path
import sys
import unittest

HAS_VISION = all(importlib.util.find_spec(name) is not None for name in ['cv2', 'numpy'])
if HAS_VISION:
    import cv2
    import numpy as np
    sys.path.insert(0, str(Path(__file__).parents[1]))
    import analyze_wrist_motion as motion


@unittest.skipUnless(HAS_VISION, 'OpenCV and NumPy required for optical analysis')
class KnownOpticalMotionTests(unittest.TestCase):
    def fixture(self, moving):
        rng = np.random.default_rng(42)
        texture = cv2.GaussianBlur(rng.integers(10, 240, (180, 240), dtype=np.uint8), (3,3), 0)
        times = np.arange(-15, 46)/30
        images = []
        for t in times:
            image = texture.copy()
            amount = round(25*np.clip((t-.15)/.4, 0, 1)) if moving else 0
            for x, sign in [(15, 1), (150, -1)]:
                image[15:75, x:x+60] = 0
                image[15:75, x+sign*amount:x+sign*amount+60] = texture[15:75, x:x+60]
            # All frames have camera shake, including the stationary baseline.
            transform = np.array([[1.,0.,2*np.sin(9*t)],[0.,1.,np.cos(7*t)]], np.float32)
            images.append(cv2.warpAffine(image, transform, (240,180)))
        motion.ROI['fixture'] = dict(closing=[[(20,20),(65,20),(65,65),(20,65)],
                                             [(155,20),(200,20),(200,65),(155,65)]],
                                     background=[[(15,105),(220,105),(220,170),(15,170)]])
        return images, times

    def test_known_onset_recovered_despite_camera_shake(self):
        images, times = self.fixture(True)
        result = motion.analyze(images, times, 'fixture', 'closing')
        self.assertTrue(result['valid'], result)
        for jaw in result['jaws']:
            lo, hi = jaw['onset_header_interval_s']
            # Detection requires >=1 px and >=2% progress, so subthreshold
            # movement can precede the reported detection interval.
            self.assertGreaterEqual(lo, .15-1/30)
            self.assertGreaterEqual(hi, .15)
            self.assertLessEqual(hi, .15+2/30)
            self.assertLessEqual(hi-lo, 1/30+1e-8)
            self.assertAlmostEqual(jaw['final_displacement_median_px'], 25, delta=.6)

    def test_camera_shake_alone_cannot_count_as_jaw_motion(self):
        images, times = self.fixture(False)
        result = motion.analyze(images, times, 'fixture', 'closing')
        self.assertFalse(result['valid'])
        self.assertIn('displacement', result['reason'])
