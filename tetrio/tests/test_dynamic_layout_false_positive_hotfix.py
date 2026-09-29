from __future__ import annotations
import unittest
import cv2
import numpy as np
from tetrio.vision.layout import OcrObservation, PlayfieldCandidate, detect_playfields, resolve_playfield_roles

def synthetic_grid(width, height, boards):
    image=np.zeros((height,width,3),dtype=np.uint8)
    for x0,y0,cell in boards:
        for c in range(11):
            x=x0+c*cell
            cv2.line(image,(x,y0),(x,y0+20*cell),(90,90,90),1)
        for r in range(21):
            y=y0+r*cell
            cv2.line(image,(x0,y),(x0+10*cell,y),(90,90,90),1)
        cv2.rectangle(image,(x0,y0),(x0+10*cell,y0+20*cell),(255,255,255),2)
    return image

class FalsePositiveHotfixTests(unittest.TestCase):
    def test_two_real_boards_survive(self):
        image=synthetic_grid(1600,900,[(250,150,26),(1000,150,26)])
        self.assertEqual(len(detect_playfields(image)),2)

    def test_single_board_does_not_ocr_score_as_username(self):
        c=PlayfieldCandidate(500,100,300,600,30,0.4,11,1.0)
        image=np.zeros((800,1200,3),dtype=np.uint8)
        resolved=resolve_playfield_roles(
            image,[c],self_username="MAYSHOWGUNMORE77",
            ocr_reader=lambda *_: OcrObservation("2,086,399",0.99),
        )
        self.assertEqual(resolved[0].role,"SELF")
        self.assertEqual(resolved[0].username,"MAYSHOWGUNMORE77")
        self.assertEqual(resolved[0].username_ocr_confidence,0.0)
        self.assertEqual(resolved[0].self_match_score,1.0)

if __name__=="__main__":
    unittest.main()
