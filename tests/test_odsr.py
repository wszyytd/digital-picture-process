import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_odsr import yolo_box, clean_name, validate_splits

class OdsrTests(unittest.TestCase):
    def test_coordinates_follow_dataset_txt_convention(self):
        self.assertEqual(yolo_box([0, 0, 640, 480], 640, 480), (.5, .5, 1., 1.))
        self.assertAlmostEqual(yolo_box([297,385,393,419],640,480)[0],345/640)

    def test_invalid_boxes_rejected(self):
        for box in ([2,1,1,5],[-1,0,5,5],[0,0,641,480],[0,0,float("nan"),5]):
            with self.assertRaises(ValueError): yolo_box(box,640,480)

    def test_correction_is_exact_and_guarded(self):
        rules={"fixes":[{"image_id":"1","object_index":0,"before":"typo","after":"stool","bbox":[1,2,3,4]}]}
        self.assertEqual(clean_name("1",0,"typo",[1,2,3,4],rules,["stool"]),"stool")
        with self.assertRaises(ValueError): clean_name("2",0,"typo",[1,2,3,4],rules,["stool"])
        with self.assertRaises(ValueError): clean_name("1",0,"typo",[1,2,3,5],rules,["stool"])

    def test_split_leakage_and_missing_ids_rejected(self):
        validate_splits({"train":["a"],"val":["b"],"test":["c"]},{"a","b","c"})
        with self.assertRaises(ValueError): validate_splits({"train":["a"],"val":["a"],"test":["c"]},{"a","b","c"})
        with self.assertRaises(ValueError): validate_splits({"train":["a"],"val":["b"],"test":[]},{"a","b","c"})
