# original code: https://github.com/ibaiGorordo/ONNX-HRNET-Human-Pose-Estimation/blob/main/HRNET/HRNET.py
# original author: Ibai Gorordo
# This code license MIT
# https://github.com/ibaiGorordo/ONNX-HRNET-Human-Pose-Estimation/blob/main/LICENSE

# modify : Toshihiko Aoki
# changes: Removed utility function dependencies to improve portability. Removed drawing related methods.

import cv2
import numpy as np
import onnxruntime


class HRNET:

    def __init__(self, path, conf_thres=0.0, search_region_ratio=0.05):
        self.conf_threshold = conf_thres
        self.search_region_ratio = search_region_ratio

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image, detections=None):

        if detections is None:
            return self.update(image)
        else:
            return self.update_with_detections(image, detections)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=['CUDAExecutionProvider',
                                                               'CPUExecutionProvider'])
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def update_with_detections(self, image, detections):

        full_height, full_width = image.shape[:2]

        boxes, scores, class_ids = detections

        if len(scores) == 0:
            self.total_heatmap, self.poses = None, None
            return self.total_heatmap, self.poses

        poses = []
        total_heatmap = np.zeros((full_height, full_width))

        for box, score in zip(boxes, scores):

            x1, y1, x2, y2 = box
            box_width, box_height = x2 - x1, y2 - y1

            # Enlarge search region
            x1 = max(int(x1 - box_width * self.search_region_ratio), 0)
            x2 = min(int(x2 + box_width * self.search_region_ratio), full_width)
            y1 = max(int(y1 - box_height * self.search_region_ratio), 0)
            y2 = min(int(y2 + box_height * self.search_region_ratio), full_height)

            crop = image[y1:y2, x1:x2]
            body_heatmap, body_pose, conf = self.update(crop)

            # Fix the body pose to the original image
            poses.append(body_pose + np.array([x1, y1]))

            # Add the heatmap to the total heatmap
            total_heatmap[y1:y2, x1:x2] += body_heatmap

        self.total_heatmap = total_heatmap
        self.poses = poses

        return self.total_heatmap, self.poses, conf

    def update(self, image):

        self.img_height, self.img_width = image.shape[:2]

        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        # Process output data
        self.total_heatmap, self.poses, conf = self.process_output(outputs)

        return self.total_heatmap, self.poses, conf

    def prepare_input(self, image):

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        input_img = ((input_img / 255.0 - mean) / std)
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor):
        # start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})[0]

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, heatmaps):
        total_heatmap = cv2.resize(heatmaps.sum(axis=1)[0], (self.img_width, self.img_height))
        map_h, map_w = heatmaps.shape[2:]

        # Find the maximum value in each of the heatmaps and its location
        max_vals = np.array([np.max(heatmap) for heatmap in heatmaps[0, ...]])
        peaks = np.array([np.unravel_index(heatmap.argmax(), heatmap.shape)
                          for heatmap in heatmaps[0, ...]])
        peaks[max_vals < self.conf_threshold] = np.array([np.NaN, np.NaN])

        # Scale peaks to the image size
        peaks = peaks[:, ::-1] * np.array([self.img_width / map_w,
                                          self.img_height / map_h])

        # add conf
        return total_heatmap, peaks, max_vals

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

