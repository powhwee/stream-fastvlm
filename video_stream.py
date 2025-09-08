#
# Modified from LLaVA/predict.py
# Please see ACKNOWLEDGEMENTS for details about LICENSE
#
import os
import argparse

import torch
from PIL import Image

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

import cv2
import threading
import subprocess as sp
import numpy as np
import sys
import queue
import time
# --- Threaded RTSP Video Capture ---
# This class reads raw video frames from stdin, which is expected to be piped
# from an ffmpeg process.
class RTSPCameraStream:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.stopped = False
        self.frame = None
        self.thread = None

        # Start the thread to read frames from stdin
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        if not self.stopped:
            self.thread.start()
        return self

    def update(self):
        frame_size = self.width * self.height * 3
        while not self.stopped:
            try:
                # Read a full frame's worth of bytes from stdin
                in_bytes = sys.stdin.buffer.read(frame_size)
                if len(in_bytes) == 0:
                    print("Input stream ended. Stopping thread.", file=sys.stderr)
                    self.stopped = True
                    break
                
                if len(in_bytes) != frame_size:
                    print(f"Warning: Incomplete frame read. Expected {frame_size}, got {len(in_bytes)}. Skipping.", file=sys.stderr)
                    continue

                # Convert the byte buffer to a numpy array and reshape it to an image
                self.frame = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])
            except Exception as e:
                print(f"Error in stdin stream update loop: {e}", file=sys.stderr)
                self.stopped = True
                break

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1) # Wait up to 1 second for the thread to finish


def inference_worker(q, model, tokenizer, image_processor, input_ids, args):
    """A worker thread that runs model inference on frames from a queue."""
    last_printed_text = ""
    while True:
        frame = q.get()
        if frame is None:  # Sentinel to stop the thread
            break

        # 1. Pre-process the Frame for the Model
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # 2. Process image for the model
        image_tensor = process_images([pil_image], image_processor, model.config)[0]

        # 3. Run inference
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half(),
                image_sizes=[pil_image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=256,
                use_cache=True)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            generated_text = outputs

        # 4. Print description to console if it has changed
        if generated_text and generated_text != last_printed_text:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"[{timestamp}] {generated_text}\n")
            last_printed_text = generated_text

        q.task_done()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a video stream from stdin with a VLM. "
                    "Example usage:\n"
                    "ffmpeg -i <rtsp_url> -f rawvideo -pix_fmt bgr24 - "
                    "| python video_stream.py --width 1280 --height 720 --model-path ...",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model-path", type=str, default="./llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--width", type=int, help="Width of the video stream from stdin.", required=True)
    parser.add_argument("--height", type=int, help="Height of the video stream from stdin.", required=True)
    parser.add_argument("--prompt", type=str, default="Describe the scene in detail.", help="Prompt for VLM.")
    parser.add_argument("--conv-mode", type=str, default="qwen_2")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    # --- Model Initialization ---
    model_path = os.path.expanduser(args.model_path)
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device="mps")

    # --- Video Stream Setup ---
    video_stream_widget = RTSPCameraStream(width=args.width, height=args.height)
    video_stream_widget.start()

    # --- Prompt Setup ---
    qs = args.prompt
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Set the pad token id for generation
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(torch.device("mps"))

    # --- Setup for Asynchronous Inference ---
    frame_queue = queue.Queue(maxsize=1)
    inference_thread = threading.Thread(
        target=inference_worker,
        args=(frame_queue, model, tokenizer, image_processor, input_ids, args),
        daemon=True
    )
    inference_thread.start()

    # --- Main Processing Loop ---
    while not video_stream_widget.stopped:
        try:
            frame = video_stream_widget.read()
            if frame is not None:
                # Non-blocking attempt to put the latest frame in the queue.
                # If the queue is full, it means inference is ongoing, so we
                # simply drop this frame and continue to display the video.
                try:
                    frame_queue.put_nowait(frame)
                except queue.Full:
                    pass

                # Display the raw frame without text overlay
                cv2.imshow("RTSP Stream with LLaVA", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except AttributeError:
            pass
        except KeyboardInterrupt:
            break

    # --- Cleanup ---
    print("Shutting down...")
    frame_queue.put(None)  # Send sentinel to stop worker thread
    inference_thread.join()
    video_stream_widget.stop()
    cv2.destroyAllWindows()
