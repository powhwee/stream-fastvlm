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
import sys
# --- Threaded RTSP Video Capture ---
# This class remains unchanged as its job is to efficiently capture
# video frames, independent of the AI framework being used.
class RTSPCameraStream:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.capture = cv2.VideoCapture(self.rtsp_url)
        if not self.capture.isOpened():
            print(f"Error: Could not open RTSP stream at {rtsp_url}", file=sys.stderr)
            self.stopped = True
            return
            
        self.stopped = False
        self.grabbed, self.frame = self.capture.read()
        if not self.grabbed:
            print("Error: Could not read initial frame from stream.", file=sys.stderr)
            self.stopped = True
            return

        # Start the thread to read frames from the video stream
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        if not self.stopped:
            self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.capture.read()
            if not self.grabbed:
                print("Stream ended or could not grab frame. Stopping thread.", file=sys.stderr)
                self.stopped = True
                break
        self.capture.release()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--rtsp-url", type=str, help="RTSP stream URL.", required=True)
    parser.add_argument("--prompt", type=str, default="Describe the scene in detail.", help="Prompt for VLM.")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
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
    video_stream_widget = RTSPCameraStream(args.rtsp_url)
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

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(torch.device("mps"))

    # --- Main Processing Loop ---
    generated_text = "Initializing..."
    last_printed_text = ""

    while not video_stream_widget.stopped:
        try:
            frame = video_stream_widget.read()
            if frame is not None:
                # 1. Pre-process the Frame for the Model
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                # 2. Process image for LLaVA
                image_tensor = process_images([pil_image], image_processor, model.config)
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)

                # 3. Run inference
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor,
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

                # 4. Print description to console and display video frame
                if generated_text and generated_text != last_printed_text:
                    print(generated_text)
                    last_printed_text = generated_text

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
    video_stream_widget.stop()
    cv2.destroyAllWindows()
