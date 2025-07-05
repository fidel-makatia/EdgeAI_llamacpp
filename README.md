# Local LLM Smart Home Assistant for SBCs

A privacy-first, locally-run smart home assistant powered by a Large Language Model (LLM) for natural language understanding and direct GPIO control on single-board computers (SBCs) such as NVIDIA Jetson and Raspberry Pi. Designed for **speed, privacy, and offline capability**—no cloud needed.

---

## 🚀 Overview

This project transforms an SBC into the intelligent core of your smart home. Unlike cloud-dependent solutions like Alexa or Google Assistant, this assistant runs a quantized 7B parameter language model *directly* on your device. It interprets complex, conversational commands (e.g., *“It’s freezing in here”* or *“I want to save on my electricity bill”*) and translates them into real-world actions—turning on heaters, switching off lights, and more—while ensuring your data never leaves your network.

---

## 🌟 Features

* **🧠 True Natural Language Understanding**
  Leverages a local LLM for context-rich, human-like command interpretation—well beyond basic keyword matching.

* **⚡ Hardware Accelerated**
  Offloads model layers to the GPU on supported NVIDIA Jetson boards for fast inference. Runs efficiently in CPU-only mode on Raspberry Pi and other SBCs.

* **🔒 100% Private & Offline**
  No internet required; all data and inference happen locally for maximum privacy and reliability.

* **🔌 Real-World GPIO Control**
  Directly interfaces with standard GPIO libraries to control relays, LEDs, fans, and other devices.

* **🚀 Hybrid System for Responsiveness**

  * **Fast Keyword Match:** Instantly handles direct commands (e.g., *“turn on kitchen light”*)
  * **LLM Inference:** Handles nuanced, ambiguous, or conversational requests with AI reasoning.

* **🌐 REST API**
  Flask-based web server for remote control and status checks via HTTP.

* **💻 Simulation Mode**
  Auto-runs in simulated GPIO mode for development on PC/Mac/Linux (no hardware needed).

---

## 🛠️ Hardware Requirements

* SBC: NVIDIA Jetson (Nano, Xavier, etc.) or Raspberry Pi (4, 5, etc.)
* MicroSD card with compatible Linux OS (Jetson Linux, Raspberry Pi OS)
* Adequate power supply
* GPIO devices (relays, LEDs, fans, etc.)
* Breadboard, jumper wires

---

## ⚡ Setup & Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2. Install Dependencies

**Create and activate a Python virtual environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### **For NVIDIA Jetson (GPU Acceleration):**

```bash
pip install llama-cpp-python[server,cuda]
pip install flask Jetson.GPIO
```

#### **For Raspberry Pi & Other SBCs (CPU-only):**

```bash
pip install llama-cpp-python[server]
pip install flask RPi.GPIO
```

### 3. Download a Quantized GGUF Model

* **Recommended:** [TheBloke/deepseek-coder-7b-instruct-GGUF](https://huggingface.co/TheBloke/deepseek-coder-7b-instruct-GGUF)
* **File:** Choose a 4-bit or 5-bit quantized model, e.g.,
  `deepseek-coder-7b-instruct.Q5_K_M.gguf`
* Place the file in a `models` directory:

```bash
mkdir models
mv ~/Downloads/deepseek-coder-7b-instruct.Q5_K_M.gguf ./models/
```

---

## 🧩 Board Adaptation

* By default, the script uses `Jetson.GPIO`.
* **On Raspberry Pi:**
  Open `rundeep.py` and replace:

  ```python
  import Jetson.GPIO as GPIO
  ```

  with

  ```python
  import RPi.GPIO as GPIO
  ```

---

## ▶️ Usage

### Running the Assistant

**For Jetson (with GPU):**

```bash
python rundeep.py --model ./models/your_model.gguf --gpu-layers 35
```

**For Raspberry Pi (CPU-only):**

```bash
python rundeep.py --model ./models/your_model.gguf --gpu-layers 0
```

#### Command-line Arguments:

* `--model` (required): Path to your GGUF model file
* `--gpu-layers`: Number of model layers to offload to GPU (`0` for CPU-only)
* `--threads`: Number of CPU threads to use (default: 4)

### Interactive Console

Once running, type commands directly in the terminal:

```
You: It's freezing in here
You: Turn off the kitchen lights
You: perf     # Shows performance stats
You: status   # Shows current device states
You: quit     # Exits
```

### Using the Flask API

* **POST command:**

  ```bash
  curl -X POST http://<sbc-ip>:5000/command \
    -H "Content-Type: application/json" \
    -d '{"text": "I want to save on my electricity bill"}'
  ```
* **GET status:**

  ```bash
  curl http://<sbc-ip>:5000/status
  ```

---

## 🛠️ Configuration

### Adding/Editing Devices

Edit the `gpio_devices` dictionary in the `SmartHomeAssistant` class within `rundeep.py`:

```python
self.gpio_devices = {
    'living_room_light': {'pin': 7, 'state': False, 'aliases': ['living room', 'main light']},
    'bedroom_light': {'pin': 11, 'state': False, 'aliases': ['bedroom']},
    # Add your device here
    'desk_fan': {'pin': 21, 'state': False, 'aliases': ['my fan', 'office fan']}
}
```

* `pin`: BOARD pin number on your SBC’s GPIO header
* `aliases`: List of alternative names the LLM can recognize

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

## 💡 Credits & Acknowledgments

Built using [llama.cpp](https://github.com/ggerganov/llama.cpp), [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), and open hardware.

---

**Questions? Issues?**
Open an issue or discussion in this repo!


