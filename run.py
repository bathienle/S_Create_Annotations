# -*- coding: utf-8 -*-

# * Copyright (c) 2022. Ba Thien LE
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
import os
import PIL
from shapely.affinity import affine_transform
from shapely.geometry import Polygon

from cytomine import Cytomine, CytomineJob
from cytomine.models import Annotation, AnnotationCollection, ImageInstance

from sldc import (
    Logger, StandardOutputLogger, Segmenter, SSLWorkflowBuilder, Tile,
    TileExtractionException, TileTopology, alpha_rasterize
)
from sldc_cytomine import CytomineSlide, CytomineTileBuilder


# Quick fix (bug with channels on Cytomine)
class ExtendedCytomineSlide(CytomineSlide):
    def __init__(self, img_instance: ImageInstance, zoom_level: int = 0):
        super(ExtendedCytomineSlide, self).__init__(img_instance, zoom_level)

    @CytomineSlide.channels.getter
    def channels(self) -> int:
        return 1


class CytominePIMSTile(Tile):
    def __init__(
        self, working_path, parent, offset, width, height,
        tile_identifier=None, polygon_mask=None
    ):
        super().__init__(
            parent, offset, width, height, tile_identifier=tile_identifier,
            polygon_mask=polygon_mask
        )

        self._working_path = working_path

    @property
    def cache_filename(self):
        image_instance = self.base_image.image_instance
        x, y = self.abs_offset_x, self.abs_offset_y
        width, height = self.width, self.height
        zoom = self.base_image.zoom_level
        return f'{image_instance.id}-{zoom}-{x}-{y}-{width}-{height}.png'

    @property
    def cache_filepath(self):
        return os.path.join(self._working_path, self.cache_filename)

    @property
    def np_image(self):
        try:
            if not os.path.exists(self.cache_filepath) and not self.download_tile_image():
                raise TileExtractionException(
                    f"Cannot fetch tile at for '{self.cache_filename}'."
                )

            np_array = np.asarray(PIL.Image.open(self.cache_filepath)).squeeze()

            if np_array.shape[:2] != (self.height, self.width) \
                    or (self.channels > 1 and (np_array.ndim < 3 or np_array.shape[2] != self.channels)) \
                    or (self.channels == 1 and np_array.ndim > 2 and np_array.shape[2] != self.channels):
                raise TileExtractionException(
                    f"Fetched image has invalid size : {np_array.shape} instead "
                    f"of {(self.width, self.height, self.channels)}"
                )

            if np_array.ndim == 3 and np_array.shape[2] == 4:
                np_array = np_array[:, :, :3]

            np_array = np_array.astype("uint8")
            return self.add_polygon_mask(np_array)
        except IOError as e:
            raise TileExtractionException(str(e))

    def add_polygon_mask(self, image):
        try:
            return alpha_rasterize(image, self.polygon_mask)
        except:
            return image

    def download_tile_image(self):
        slide = self.base_image
        filepath = slide.image_instance.path
        topology = TileTopology(slide, None, max_width=256, max_height=256)
        col_tile = self.abs_offset_x // 256
        row_tile = self.abs_offset_y // 256
        tile_index = col_tile + row_tile * topology.tile_horizontal_count
        _slice = slide.slice_instance

        url = f'{_slice.imageServerUrl}/image/{filepath}/tile/' \
              f'zoom/{slide.api_zoom_level}/ti/{tile_index}.png'

        return Cytomine.get_instance().download_file(url, self.cache_filepath)


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
        heatmap = ExtendedCytomineSlide(ImageInstance().fetch(cj.parameters.heatmap_id))

        # Build the workflow
        cj.job.update(progress=20, statusComment="Build the workflow")
        builder = SSLWorkflowBuilder()
        builder.set_background_class(0)
        builder.set_tile_size(512, 512)
        builder.set_tile_builder(
            CytomineTileBuilder(working_path, tile_class=CytominePIMSTile)
        )
        builder.set_logger(StandardOutputLogger(level=Logger.WARNING))
        builder.set_segmenter(ThresholdSegmenter(cj.parameters.threshold))

        # Get the workflow
        workflow = builder.get()

        # Process the image
        cj.job.update(progress=40, statusComment="Apply the threshold")
        results = workflow.process(heatmap)
        annotations = AnnotationCollection()

        cj.job.update(progress=80, statusComment="Save the annotations")
        for result in results:
            if check_area(result.polygon, min_area=cj.parameters.min_area):
                annotation = affine_transform(
                    result.polygon,
                    [1, 0, 0, -1, 0, heatmap.height]
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
