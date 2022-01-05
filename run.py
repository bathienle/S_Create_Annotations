# -*- coding: utf-8 -*-

# * Copyright (c) 2021. Ba Thien LE
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.

import numpy as np
import sys
from argparse import ArgumentParser
from shapely.affinity import affine_transform
from shapely.geometry import Polygon

from cytomine import Cytomine
from cytomine.models import Annotation, AnnotationCollection, ImageInstance

from sldc import Logger, StandardOutputLogger, Segmenter, SSLWorkflowBuilder
from sldc_cytomine import CytomineSlide, CytomineTileBuilder


# Quick fix (bug with channels on Cytomine)
class ExtendedCytomineSlide(CytomineSlide):
    def __init__(self, img_instance: ImageInstance, zoom_level: int = 0):
        super(ExtendedCytomineSlide, self).__init__(img_instance, zoom_level)

    @CytomineSlide.channels.getter
    def channels(self) -> int:
        return 1


class ThresholdSegmenter(Segmenter):
    def __init__(self, threshold: float = 0.5):
        super(ThresholdSegmenter, self).__init__()

        self.threshold = int(threshold * 255)

    def segment(self, image: np.array) -> np.array:
        mask = (image > self.threshold).astype(np.uint8)
        mask[mask == 1] = 255
        return mask


def check_area(polygon: Polygon, min_area: int) -> bool:
    return min_area < polygon.area


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument(
        '--host',
        help="The Cytomine host"
    )
    parser.add_argument(
        '--public_key',
        help="The Cytomine public key"
    )
    parser.add_argument(
        '--private_key',
        help="The Cytomine private key"
    )
    parser.add_argument(
        '--project_id',
        help="The project from which we want the images"
    )
    parser.add_argument(
        '--image_id',
        help="The image from which to generate the annotations"
    )
    parser.add_argument(
        '--term_id',
        help="The term ID to associate to the annotations"
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help="The threshold for the segmentation"
    )
    parser.add_argument(
        '--min_area',
        type=int,
        default=1000,
        help="The minimum area for an annotation"
    )

    args, _ = parser.parse_known_args(sys.argv[1:])
    working_path = 'tiles/'

    with Cytomine(args.host, args.public_key, args.private_key) as cytomine:
        image = ExtendedCytomineSlide(ImageInstance().fetch(args.image_id))

        # Build the workflow
        builder = SSLWorkflowBuilder()
        builder.set_tile_size(512, 512)
        builder.set_tile_builder(CytomineTileBuilder(working_path))
        builder.set_logger(StandardOutputLogger(level=Logger.INFO))
        builder.set_segmenter(ThresholdSegmenter(args.threshold))

        # Get the workflow
        workflow = builder.get()

        # Process the image
        results = workflow.process(image)
        annotations = AnnotationCollection()
        for result in results:
            if check_area(result.polygon, min_area=args.min_area):
                annotation = affine_transform(
                    result.polygon,
                    [1, 0, 0, -1, 0, image.height]
                )

                annotations.append(Annotation(
                    location=annotation.wkt,
                    id_image=args.image_id,
                    id_project=args.project_id,
                    term=[args.term_id]
                ))

        annotations.save()
