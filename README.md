# Stream-VLM: Stream video to Apple's FastVLM using ffmpeg for captioning / descriptions

Apple's FastVLM repository (https://github.com/apple/ml-fastvlm) comes with a demo predict.py that works with an image, and also iOS/MacOS apps to demo its features.

This project uses ffmpeg to pipe video, images, including video streams to FastVLM for interpretation without using the 'predict.py' or the Swift apps.  The output is then logged with timestamp. This is with the view that logs can be more easily searched, analysed and perhaps even summarised by text LLM.  

Combining FastVLM with an LLM also allow filtering out unique scenes, as illustrated in the example below.


## Prerequisites

-   Python 3.10
-   FFmpeg

Note: Python 3.13 didn't work when I tested it.  

## Setup

1.  **Create and activate a virtual environment:**
    ```bash
    python3.10 -m venv venv
    # if you have multiple python versions on your machine, use the one that works for FastVLM to create the venv
    source venv/bin/activate
    ```
    

2.  **Clone the repository:**
    ```bash
    git clone https://github.com/powhwee/stream-fastvlm.git
    cd stream-fastvlm
    pip install -r requirements.txt
    # the requirements.txt is built from requirement.in, which does an include of the .toml file that comes with Apple's FastVLM
    ```


3.  **Download Model Weights:**
    ```bash
    # Assuming you are in the stream-fastvlm directory, chmod u+x this .sh file and run it to download Apple's 0.5b model file.  There are model with 1.5b and 7b parameters but this script downloads only the 0.5b model
    cd checkpoints  
    chmod u+x get_0.5b_model.sh  
    ./get_0.5b_model.sh
    ```
    
    Download FastVLM's model and place it in a known directory. The example command below points to local folders (e.g., `./checkpoints/llava-fastvithd_0.5b_stage3`), so adjust the path as needed.

## Usage


1.  **Find your RTSP stream URL.** This is specific to your camera or video source.
2.  **Determine the stream's resolution (width and height).** You can often find this in your camera's documentation or by using tools like `ffprobe`.
3.  **Run the application** using the following command structure:

    ```bash
    ffmpeg -i "rtsp://<YOUR_RTSP_URL>" -f rawvideo -pix_fmt bgr24 - \
    | python video_stream.py --width <STREAM_WIDTH> --height <STREAM_HEIGHT> --model-path <PATH_TO_YOUR_MODEL>

    # RTSP_URL typically looks like this : rtsp://<id>:<your_password>@<ip address>:554/<streamId>```

### Example

Example of using a RTSP URL and a 1280x720 resolution stream:

```bash
ffmpeg -i "rtsp://rtspsourcename:credential@192.168.1.20/stream2" -f rawvideo -pix_fmt bgr24 - \
| python video_stream.py --width 1280 --height 720 --model-path ./checkpoints/llava-fastvithd_0.5b_stage3 --prompt "Describe this scene."
```


The script will open a window displaying the video feed, and you will see timestamped descriptions appearing in your console. Press `q` with the video window in focus to quit.

#### Test video

For ease of verifying your setup before you get to RTSP, I used the following test video.  Note the prompt I use to give the model a bit of challenge to find out what is unique.

```bash
ffmpeg -re -i "test_data/test_video_chicken_cross_the_road.mov" -f rawvideo -pix_fmt bgr24 - | python video_stream.py --width 1280 --height 1064 --prompt "Describe what is unique about this scene" --model-path ./checkpoints/llava-fastvithd_0.5b_stage3
```

<img src="test_data/test_image_chicken_cross_the_road.png" width="640" height="auto" />


These are the outputs, which is interesting that the VLM is able to identify the uniqueness of the scene:

`[2025-09-10 20:57:38] The scene is unique because it captures a moment where a rooster is seen in the foreground, seemingly walking across a pedestrian crossing, while a car is in motion in the background. The combination of the rooster and the car creates a striking contrast between the natural and the man-made elements in the image. The rooster's vibrant colors and the motion blur of the car add a dynamic quality to the scene, suggesting a moment of unexpected interaction between the two.`

`[2025-09-10 20:57:41] The scene is unique due to the presence of a rooster walking across the street, which is not a common sight in many urban areas. The rooster's vibrant red color contrasts sharply with the muted tones of the surrounding buildings and the grey sky. Additionally, the street is lined with parked cars and a few pedestrians, suggesting a typical day in a bustling city. The juxtaposition of the rooster against the backdrop of modern buildings and the urban environment creates a striking image that captures the essence of city life.`

`[2025-09-10 20:57:42] This scene is unique because it features a chicken walking across a city street, which is an unusual sight in an urban environment. The presence of the chicken amidst the traffic and buildings is unexpected and adds a whimsical element to the otherwise typical cityscape.`



### Command-Line Arguments
`python video_stream.py`
-   `--model-path`: (Required) Path to the VLM checkpoint.
-   `--width`: (Required) Width of the input video stream.
-   `--height`: (Required) Height of the input video stream.
-   `--prompt`: The prompt to use for generating descriptions. Default: `"Describe the scene in detail."`
-   `--conv-mode`: The conversation template to use (e.g., `qwen_2`, `llava_v1`). Default: `"qwen_2"`.
-   `--temperature`, `--top_p`, `--num_beams`: Standard model generation parameters.

## Acknowledgements

This project utilizes components and concepts from the LLaVA (Large Language and Vision Assistant) project. Please refer to their repository for details on their license and original work.

