from .segformer_b2_clothes import *

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "segformer_b2_clothes":segformer_b2_clothes
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "segformer_b2_clothes":"segformer_b2_clothes"
}
