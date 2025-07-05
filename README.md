# Local LLM Smart Home Assistant for Arm-Based SBCs

A **powerful, private, and offline** smart home assistant built to showcase the true potential of Arm architecture. By leveraging a quantized Large Language Model (LLM), this project demonstrates how modern Arm-based single-board computers—such as NVIDIA Jetson (Arm Cortex-A57, A78AE, etc.) and Raspberry Pi (Arm Cortex-A72, A76)—can deliver state-of-the-art AI functionality and real-world device control, right at the edge.

---

## 🚀 Why Arm? (And Why This Project?)

**Arm architecture** powers billions of intelligent edge devices thanks to its unique blend of:

* **Exceptional energy efficiency** (run AI at a fraction of the power of x86/desktop-class hardware)
* **Scalable performance** (from tiny microcontrollers to high-performance SBCs and SoCs)
* **World-class support for AI and ML acceleration** (NEON, GPU, NPU/AI co-processors, and more)
* **Open ecosystem** with robust Python, Linux, and GPIO support

This assistant puts these strengths on full display by running a full 7B-parameter LLM (in GGUF format) locally on Arm-powered SBCs, combining real-time natural language understanding, direct GPIO control, and a hybrid fast/AI decision pipeline—all with no cloud dependency and **no privacy compromise**.

---

## 🌟 Project Highlights

* **🧠 Edge AI on Arm:**
  Runs advanced LLM inference directly on affordable Arm SBCs, transforming them into voice-controllable smart home hubs.

* **⚡ Arm-Optimized Acceleration:**
  Leverages on-chip GPUs (like NVIDIA Jetson’s CUDA-capable GPU) for rapid model inference, with smooth CPU-only operation on Raspberry Pi and other Cortex-A platforms.

* **🔋 True Efficiency:**
  AI at the edge, with minimal power draw—run 7B-parameter models on a device powered via USB-C or a simple wall adapter.

* **🔒 Complete Local Privacy:**
  All voice/text commands are processed and reasoned about locally—no data leaves your network, unlike traditional cloud assistants.

* **🔌 Direct Hardware Control:**
  Native integration with Arm Linux GPIO libraries for real-world automation—relays, fans, lights, and more.

* **🌐 REST API Built-In:**
  Flask-powered web API for remote network control from any browser or mobile device.

---

## 🛠️ Hardware Requirements

* **Arm-based SBC:**

  * **NVIDIA Jetson** (Nano, Xavier, Orin, etc. – Arm Cortex-A57/A78AE, GPU acceleration)
  * **Raspberry Pi 4/5** (Arm Cortex-A72/A76, CPU-only)
  * Other compatible Arm Linux boards (with GPIO and Python3)
* **MicroSD card** (with Jetson Linux or Raspberry Pi OS)
* **Power supply** for your board
* **GPIO devices**: relays, LEDs, fans, etc.

---

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2. Install Python Dependencies

**Set up a Python 3 virtual environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### On Jetson (GPU/Arm):

```bash
pip install llama-cpp-python[server,cuda]
pip install flask Jetson.GPIO
```

#### On Raspberry Pi (CPU/Arm):

```bash
pip install llama-cpp-python[server]
pip install flask RPi.GPIO
```

### 3. Download a Quantized GGUF LLM

* **Recommended:** [TheBloke/deepseek-coder-7b-instruct-GGUF](https://huggingface.co/TheBloke/deepseek-coder-7b-instruct-GGUF)
* Download a 4-bit or 5-bit model, e.g.:
  `deepseek-coder-7b-instruct.Q5_K_M.gguf`
* Place in a `models` directory:

```bash
mkdir models
mv ~/Downloads/deepseek-coder-7b-instruct.Q5_K_M.gguf ./models/
```

---

## ▶️ Running the Assistant on Arm

**For Jetson (Arm + GPU):**

```bash
python rundeep.py --model ./models/your_model.gguf --gpu-layers 35
```

**For Raspberry Pi (Arm CPU-only):**

```bash
python rundeep.py --model ./models/your_model.gguf --gpu-layers 0
```

* `--model`: Path to your GGUF model
* `--gpu-layers`: Number of model layers to accelerate with GPU (set to 0 on Pi)

---

## 💡 What Makes Arm Shine Here?

* **Low-Power, High-Performance AI**:
  Run a billion-parameter LLM while using a tiny fraction of the energy of x86/desktop hardware.
* **Compact, Scalable Hardware**:
  Everything fits on a board that’s smaller than your phone, and can run from a battery pack.
* **Open, Flexible Ecosystem**:
  No vendor lock-in—adaptable to any Arm Linux board with standard Python and GPIO support.
* **Direct Access to Hardware Accelerators**:
  Exploit Jetson’s CUDA cores, NPU/AI accelerators, or run on pure CPU for broad compatibility.
* **Empowering Next-Gen Edge AI**:
  Move sophisticated AI out of the cloud, onto the device—enabling privacy, real-time control, and zero latency for the end user.

---

## 🛠️ Device Customization

Edit the `gpio_devices` dictionary in `rundeep.py` to add or modify your GPIO-controlled devices (pins, aliases, initial state).

```python
self.gpio_devices = {
    'living_room_light': {'pin': 7, 'state': False, 'aliases': ['living room', 'main light']},
    'bedroom_light': {'pin': 11, 'state': False, 'aliases': ['bedroom']},
    # More devices...
}
```

---

## 🌐 API & Console

* **Interactive Console**:
  Type commands like:
  `It's freezing in here`
  `Turn off the kitchen light`
  `status` (shows device states)
  `perf` (shows inference stats)
  `quit` (exit)

* **Flask REST API**:
  POST commands to `http://<sbc-ip>:5000/command`
  GET device status from `http://<sbc-ip>:5000/status`

---

## 📝 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🎯 For Arm Innovators

This project is a proof-of-concept that **Arm architecture is ready for the AI-powered future of edge devices**.
Deploy advanced LLMs. Run them fast and efficiently. Keep data private and on-device.
**This is the kind of real-world use case that only Arm can deliver at scale and efficiency.**

