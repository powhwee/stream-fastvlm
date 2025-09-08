import cv2
import threading
import sys
from PIL import Image

# Import the high-level functions from the mlx-vlm library
# This replaces the need for AutoTokenizer and AutoModelForCausalLM from transformers
from mlx_vlm import load, generate

# --- 1. AI Model Setup (MLX Version) ---
# This section is now much simpler. The `load` function handles all the
# complex setup of the model, tokenizer, and processor automatically.
def initialize_mlx_pipeline(model_id: str = "apple/FastVLM-1.5B-int8"):
    """
    Loads the FastVLM model and processor using the mlx-vlm library.
    MLX automatically uses the GPU and Neural Engine on Apple Silicon.
    """
    print(f"Initializing MLX pipeline with model: {model_id}", file=sys.stderr)
    try:
        # A single function call loads everything needed for inference.
        model, processor = load(model_id)
        print("MLX model loaded successfully.", file=sys.stderr)
        return model, processor
    except Exception as e:
        print(f"Failed to load MLX model: {e}", file=sys.stderr)
        sys.exit(1)

# --- 2. Threaded RTSP Video Capture ---
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

# --- 3. Main Execution Block ---
if __name__ == '__main__':
    # --- Model Initialization ---
    # Choose the MLX-compatible model from Hugging Face
    model_id = "apple/FastVLM-1.5B-int8"
    model, processor = initialize_mlx_pipeline(model_id)

    # --- Video Stream Setup ---
    rtsp_stream_link = "rtsp://your_user:your_password@camera_ip:554/stream_path"
    video_stream_widget = RTSPCameraStream(rtsp_stream_link)
    video_stream_widget.start()

    # --- Main Processing Loop ---
    prompt = "Describe the scene in detail."
    generated_text = "Initializing..."

    while not video_stream_widget.stopped:
        try:
            frame = video_stream_widget.read()
            if frame is not None:
                
                # =================================================================
                # --- START OF FASTVLM INFERENCE WORKFLOW (MLX Version) ---
                # This block is the MLX equivalent of the original script's
                # manual prompt preparation and model.generate() call.
                # =================================================================

                # 1. Pre-process the Frame for the Model
                # OpenCV uses BGR format; convert to RGB for the model.
                # The mlx-vlm library expects a PIL Image.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                # 2. Generate Description using FastVLM
                # This single, high-level function replaces all the manual
                # tokenizing, splitting, and tensor concatenation.
                generated_text = generate(model, processor, pil_image, prompt)

                # =================================================================
                # --- END OF FASTVLM INFERENCE WORKFLOW ---
                # =================================================================

                # 3. Display the Video and Description
                # Overlay the generated text onto the video frame.
                # We add a semi-transparent background for better readability.
                text_y_pos = 30
                (w, h), _ = cv2.getTextSize(generated_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (0, text_y_pos - 20), (w + 10, text_y_pos + 10), (0,0,0), -1)
                cv2.putText(frame, generated_text, (5, text_y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("RTSP Stream with MLX FastVLM", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except AttributeError:
            # This can happen if the stream is starting up or has an issue.
            pass
        except KeyboardInterrupt:
            break

    # --- Cleanup ---
    print("Shutting down...")
    video_stream_widget.stop()
    cv2.destroyAllWindows()
