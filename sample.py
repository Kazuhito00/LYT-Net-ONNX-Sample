#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse
from typing import Any

import cv2
import numpy as np
import onnxruntime  # type: ignore


def run_inference(
    onnx_session,
    image: np.ndarray,
) -> np.ndarray:
    input_detail = onnx_session.get_inputs()[0]
    input_name: str = input_detail.name
    input_shape = input_detail.shape[1:3]

    # Pre process: Resize, Normalize, float32 cast, Transpose
    input_image: np.ndarray = cv2.resize(
        image,
        dsize=(input_shape[1], input_shape[0]),
    )
    input_image = (input_image / 255.0) * 2 - 1.0
    input_image = input_image.astype('float32')
    input_image = input_image.reshape(1, *input_shape, 3)

    # Inference
    result: np.ndarray = onnx_session.run(None, {input_name: input_image})[0]

    # Post process: squeeze, uint8 cast, Resize
    result = np.array(result).squeeze()
    output_image: np.ndarray = (((result + 1.0) / 2.0) * 255).astype('uint8')
    output_image = cv2.resize(
        output_image,
        dsize=(image.shape[1], image.shape[0]),
    )

    return output_image


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default='model/lyt_net_lolv2_real_320x240.onnx',
    )

    args = parser.parse_args()
    model_path: str = args.model

    # Initialize video capture
    cap_device: Any = args.device
    if args.movie is not None:
        cap_device = args.movie
    cap: cv2.VideoCapture = cv2.VideoCapture(cap_device)

    # Load model
    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )

    while True:
        start_time = time.time()

        # Capture read
        ret: bool
        frame: np.ndarray
        ret, frame = cap.read()
        if not ret:
            break
        debug_image: np.ndarray = copy.deepcopy(frame)

        # Inference execution
        output_image: np.ndarray = run_inference(
            onnx_session,
            frame,
        )

        elapsed_time = time.time() - start_time

        # Inference elapsed time
        cv2.putText(
            debug_image,
            "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1,
            cv2.LINE_AA)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        cv2.imshow('LYT-Net Input', debug_image)
        cv2.imshow('LYT-Net Output', output_image)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
