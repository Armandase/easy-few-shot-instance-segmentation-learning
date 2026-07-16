from pathlib import Path
from typing import Callable, Optional, Union

from .wrap_few_shot_dataset import WrapFewShotDataset

COCO_IMGS_DIR = Path("data/coco_imgs")
COCO_ANNS_FILE = Path("data/coco_anns/captions_val2017.json")


import torchvision.datasets as dset
import torchvision.transforms as transforms

def get_coco_set(coco_img_dir: Union[Path, str] = COCO_IMGS_DIR,
                 coco_ann_file: Union[Path, str] = COCO_ANNS_FILE,
                 transform: Optional[Callable] = [transforms.ToTensor()],):
    print("coco_img_dir", coco_img_dir)
    print("coco_ann_file", coco_ann_file)
    cap = dset.CocoCaptions(root = coco_img_dir,
                        annFile = coco_ann_file,
                        transform=transform)
    
    coco = WrapFewShotDataset(
            cap,
            image_position_in_get_item_output=0,
            label_position_in_get_item_output=1,
        )
    return coco


if __name__ == "__main__":
    get_coco_set("/home/adamiens/coco/coco2017/coco2017/train2017/", "/home/adamiens/coco/coco2017/coco2017/annotations/instances_train2017.json")