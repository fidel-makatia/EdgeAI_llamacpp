Local LLM Smart Home Assistant for SBCs
A powerful, locally-run smart home assistant that uses a Large Language Model (LLM) to understand natural language and control GPIO-connected devices on single-board computers (SBCs) like the NVIDIA Jetson and Raspberry Pi. This project is designed for privacy, speed, and offline capability.

Overview
This project turns an SBC into the brain of a smart home. Instead of relying on cloud services like Alexa or Google Assistant, it runs a quantized 7B parameter language model directly on the device. This allows it to interpret complex, conversational commands (e.g., "It's freezing in here" or "I want to save on my electricity bill") and translate them into actions, like turning on a heater or switching off all lights.

Features
🧠 True Natural Language Understanding: Leverages a local LLM to go beyond simple keyword matching.

⚡ Hardware Accelerated: Offloads model layers to the GPU on supported NVIDIA hardware for significantly faster inference. Runs efficiently in CPU-only mode on other devices.

🔒 Private & Offline: No data ever leaves your local network. It works perfectly without an internet connection.

🔌 Real-World Control: Directly interfaces with standard GPIO libraries to control relays, LEDs, fans, and other electronic components.

🚀 Hybrid System: Uses a tiered approach for optimal performance:

Fast Keyword Match: Instantly handles simple, direct commands ("turn on the kitchen light").

LLM Inference: Consults the AI for complex, conversational, or ambiguous requests.

🌐 Web API: Includes a Flask-based web server to control the assistant remotely over your network.

💻 Simulation Mode: Automatically runs in a simulated GPIO mode for easy development and testing on a Mac, Windows, or Linux PC without GPIO hardware.

How It Works
The assistant follows a logical flow to interpret and execute commands:

Input: Receives a command from the user via the command line or the Flask API.

Keyword Check: First, it checks for a simple, direct command (e.g., "turn on fan"). If a high-confidence match is found, it executes immediately for maximum speed.

LLM Inference: If the command is more complex, it's sent to the LLM. A detailed system prompt provides the model with the current context (time of day, device states) to help it make an intelligent decision.

Structured JSON Output: The LLM's task is to return a structured JSON object containing the intent, target devices, and its reasoning.

Execution: The main script parses the JSON and calls the appropriate function to control the GPIO pins.

Feedback: The assistant reports back the action it took.

Hardware Requirements
A single-board computer (SBC) like an NVIDIA Jetson (Nano, Xavier) or Raspberry Pi (4, 5).

MicroSD card with a compatible Linux OS (Jetson Linux, Raspberry Pi OS).

Adequate power supply for your board.

GPIO devices (e.g., relay modules, LEDs).

Breadboard and jumper wires.

Setup & Installation

1. Clone the Repository
   git clone <your-repo-url>
   cd <your-repo-url>

2. Install Dependencies
   It's highly recommended to use a Python virtual environment.

# Create and activate a virtual environment

python3 -m venv .venv
source .venv/bin/activate

For NVIDIA Jetson (GPU Acceleration)

# Install llama-cpp-python with CUDA support

# This is the most important step for performance on Jetson!

pip install llama-cpp-python[server,cuda]

# Install other required packages

pip install flask Jetson.GPIO

For Raspberry Pi & Other SBCs (CPU-Only)

# Install llama-cpp-python without GPU support

pip install llama-cpp-python[server]

# Install other required packages

# Note: You may need to install RPi.GPIO or another library (see Board Adaptation below)

pip install flask RPi.GPIO

3. Download a GGUF Model
   This project requires a model in the GGUF format. For a 7B model, it's crucial to use a quantized version to ensure it runs efficiently.

Recommended Model: TheBloke/deepseek-coder-7b-instruct-GGUF

File to Download: A 4-bit or 5-bit quantized model like deepseek-coder-7b-instruct.Q4_K_M.gguf or deepseek-coder-7b-instruct.Q5_K_M.gguf is a great balance of performance and quality.

Create a models directory and place the downloaded .gguf file inside it.

mkdir models
mv ~/Downloads/deepseek-coder-7b-instruct.Q5_K_M.gguf ./models/

Board Adaptation
The provided Python script uses the Jetson.GPIO library by default. If you are using a different board, you will need to make a small change to the code.

For Raspberry Pi:

Open the main Python script.

Find the line import Jetson.GPIO as GPIO.

Replace it with import RPi.GPIO as GPIO.

The rest of the code, which uses the GPIO alias, should work without changes.

Usage
Running the Assistant
Execute the main Python script from your terminal.

# For Jetson (with GPU)

python your_script_name.py --model ./models/your_model.gguf --gpu-layers 35

# For Raspberry Pi (CPU-only)

python your_script_name.py --model ./models/your_model.gguf --gpu-layers 0

Command-Line Arguments:
--model: (Required) Path to your GGUF model file.

--gpu-layers: Number of model layers to offload to the GPU. Set to 0 for Raspberry Pi. A higher number improves performance on Jetson.

--threads: Number of CPU threads to use.

Interactive Console
Once running, you can type commands directly into the terminal:

You: It's freezing in here

You: Turn off the living room and kitchen lights

You: perf (Shows performance statistics)

You: status (Shows the current state of all devices)

You: quit (Exits the program)

Using the Flask API
The assistant also exposes an API on port 5000. You can interact with it from any device on the same network.

Send a command:

curl -X POST http://<sbc-ip>:5000/command \
 -H "Content-Type: application/json" \
 -d '{"text": "I want to save on my electricity bill"}'

Check status:

curl http://<sbc-ip>:5000/status

Configuration
Adding Your Devices
To add or change your GPIO devices, simply edit the gpio_devices dictionary in the SmartHomeAssistant class **init** method.

self.gpio_devices = {
'living_room_light': {'pin': 7, 'state': False, 'aliases': ['living room', 'main light']},
'bedroom_light': {'pin': 11, 'state': False, 'aliases': ['bedroom']}, # Add your new device here
'desk_fan': {'pin': 21, 'state': False, 'aliases': ['my fan', 'office fan']}
}

pin: The BOARD pin number on your SBC's GPIO header.

aliases: A list of alternative names the LLM can use to identify the device.

License
This project is licensed under the MIT License. See the LICENSE file for details.
