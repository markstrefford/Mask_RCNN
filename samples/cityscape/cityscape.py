"""
Mask R-CNN
Train on the toy CityScape dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 cityscape.py train --dataset=/path/to/cityScape/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 cityscape.py train --dataset=/path/to/cityScape/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 cityscape.py train --dataset=/path/to/cityScape/dataset --weights=imagenet

    # Apply color splash to an image
    python3 cityscape.py genmasks --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 cityscape.py genmasks --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from skimage import img_as_uint
from pathlib import Path

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class CityScapeConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "cityscape"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + cityscape

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class CityScapeDataset(utils.Dataset):

    # TODO - Find a way to add class_list to __init__() without overriding utils.Dataset __init__()!
    # TODO - Determine how best to handle multiple classes here. Currently just using building.
    #  Issue arose in displaying the masks as it defaulted to the 1st class (road)
    def load_cityscape(self, dataset_dir, mask_dir, subset):
        """Load a subset of the CityScape dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes.
        # From https://www.cityscapes-dataset.com/dataset-overview/#features
        cityscape_classes = [
            'building'
        ]
        #         cityscape_classes = [
        #             'road', 'sidewalk', 'parking', 'rail', 'track',
        #             'person', 'rider',
        #             'car', 'truck', 'bus', 'on rails', 'motorcycle', 'bicycle', 'caravan', 'trailer',
        #             'building', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
        #             'pole', 'pole group', 'traffic sign', 'traffic light',
        #             'vegetation', 'terrain',
        #             'ground', 'dynamic', 'static'
        #         ]
        for id, cityscape_class in enumerate(cityscape_classes):
            self.add_class(cityscape_class, id, cityscape_class)

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        mask_dir = os.path.join(mask_dir, subset)

        image_list = Path(dataset_dir).glob('**/*.png')
        for image_path in image_list:
            city, image_file = str(image_path).split(os.sep)[-2:]

            # Get JSON file
            json_file = image_file.replace('_leftImg8bit.png', '_gtFine_polygons.json')
            json_filepath = os.path.join(mask_dir, city, json_file)

            # Load mask polygons json
            # From https://stackoverflow.com/a/55016816/1378071 as cityscapes json wouldn't load without this!
            with open(json_filepath, encoding='utf-8', errors='ignore') as json_data:
                mask_json = json.load(json_data, strict=False)

            h, w = mask_json['imgHeight'], mask_json['imgWidth']

            # Get masks for each object
            objects = list(mask_json['objects'])

            polygons = []
            for object in objects:
                obj_class = object['label']
                obj_polygons = object['polygon']
                if obj_class == 'building' and obj_polygons != []:
                    polygon = dict()
                    all_points_y, all_points_x = [], []
                    for x, y in obj_polygons:
                        # Handle polygons outside of image area
                        x = x if x < w else w - 1
                        y = y if y < h else h - 1
                        all_points_x.append(x)
                        all_points_y.append(y)
                        polygon['all_points_y'] = all_points_y
                        polygon['all_points_x'] = all_points_x
                    polygons.append(polygon)

            self.add_image(
                'building',
                image_id=image_file,
                path=image_path,
                width=w, height=h,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a cityscape dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "building":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "building":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CityScapeDataset()
    dataset_dir = os.path.join(args.dataset, 'leftImg8bit_trainvaltest/leftImg8bit')
    mask_dir = os.path.join(args.dataset, 'gtFine_trainvaltest/gtFine')
    dataset_train.load_cityscape(dataset_dir, mask_dir, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CityScapeDataset()
    dataset_val.load_cityscape(dataset_dir, mask_dir, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=4,    # Low number of epochs due to large training set size (5000+ images)
                layers='heads')


def detect_and_create_mask(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate masks
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        print('Image {} - Found {} masks'.format(image_path, r['masks'].shape))
        # Save output
        img_dir = os.path.dirname(image_path) + '_seg'
        img_file = os.path.basename(image_path)
        # All masks as one for now... (but really check size of r[-1] as this is number of masks...
        # for i, mask in enumerate(r['masks']):
        mask = r['masks']
        if mask.shape[-1] > 0:
            i = 0
            fname = '{}_{}.{}'.format(img_file.split('.')[0], i, img_file.split('.')[1])
            file_name = os.path.join(img_dir, fname)
            print('Saving mask to {}'.format(file_name))
            skimage.io.imsave(file_name, img_as_uint(r['masks']))
        else:
            # TODO: Generate blank mask and save to not confuse training of GAN later...!
            print('No masks generated!')
    elif video_path:
        # TODO: Modify video code as above
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect cityscapes.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'mask'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/cityscape/dataset/",
                        help='Directory of the CityScape dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "genmask":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CityScapeConfig()
    else:
        class InferenceConfig(CityScapeConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "genmask":
        detect_and_create_mask(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'genmask'".format(args.command))
