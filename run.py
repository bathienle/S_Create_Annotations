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
from shapely.affinity import affine_transform
from shapely.geometry import Polygon

from cytomine import CytomineJob
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


def main(argv):
    working_path = 'tiles/'

    with CytomineJob.from_cli(argv) as cj:
        cj.job.update(progress=0, statusComment="Fetch the image from Cytomine")
        image = ExtendedCytomineSlide(ImageInstance().fetch(cj.parameters.image_id))

        # Build the workflow
        cj.job.update(progress=20, statusComment="Build the workflow")
        builder = SSLWorkflowBuilder()
        builder.set_tile_size(512, 512)
        builder.set_tile_builder(CytomineTileBuilder(working_path))
        builder.set_logger(StandardOutputLogger(level=Logger.WARNING))
        builder.set_segmenter(ThresholdSegmenter(cj.parameters.threshold))

        # Get the workflow
        workflow = builder.get()

        # Process the image
        cj.job.update(progress=40, statusComment="Apply the threshold")
        results = workflow.process(image)
        annotations = AnnotationCollection()

        cj.job.update(progress=80, statusComment="Save the annotations")
        for result in results:
            if check_area(result.polygon, min_area=cj.parameters.min_area):
                annotation = affine_transform(
                    result.polygon,
                    [1, 0, 0, -1, 0, image.height]
                )

                annotations.append(Annotation(
                    location=annotation.wkt,
                    id_image=cj.parameters.image_id,
                    id_project=cj.parameters.cytomine_id_project,
                    term=list(map(int, cj.parameters.cytomine_term_ids.split(',')))
                ))

        annotations.save()

        cj.job.update(progress=100, statusComment="Job terminated")


if __name__ == '__main__':
    import sys

    main(sys.argv[1:])
