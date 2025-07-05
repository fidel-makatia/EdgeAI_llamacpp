#!/usr/bin/env python3
import json
import time
import re
import argparse
import threading
from datetime import datetime
from collections import deque
import statistics
from llama_cpp import Llama
import Jetson.GPIO as GPIO
from flask import Flask, request, jsonify

class SmartHomeAssistant:
    """
    A smart home assistant that uses a local LLM for natural language understanding
    and controls GPIO devices on a Jetson Nano. It features a tiered system:
    1. Fast keyword matching for simple, direct commands.
    2. LLM-based interpretation for complex, implicit commands.
    3. Robust JSON parsing to handle imperfect model outputs.
    """
    def __init__(self, model_path, n_gpu_layers=35, n_threads=4):
        """
        Initializes the assistant, loads the LLM, and sets up GPIO.

        Args:
            model_path (str): The path to the GGUF model file.
            n_gpu_layers (int): Number of layers to offload to GPU. Set to 0 for CPU-only.
            n_threads (int): The number of CPU threads to use for inference.
        """
        print("Initializing Smart Home Assistant...")
        self._setup_performance_tracking()

        # --- LLM Loading ---
        # This section loads the language model from the specified file path.
        # It uses parameters optimized for a balance of speed and coherence on embedded hardware.
        print(f"Loading model from: {model_path}")
        load_start = time.time()
        self.llm = Llama(
            model_path=model_path,
            n_ctx=1024,          # The maximum context size (in tokens) the model can handle.
            n_threads=n_threads, # Number of CPU threads to use for generation.
            n_gpu_layers=n_gpu_layers, # The number of layers to offload to the GPU. This is the most important parameter for performance.
            n_batch=256,         # The number of tokens to process in parallel.
            use_mmap=True,       # Use memory mapping to load the model faster.
            f16_kv=True,         # Use 16-bit precision for the key/value cache to save memory.
            seed=-1,             # A random seed for reproducibility. -1 means random.
            verbose=False,       # Suppress detailed, low-level logs from the llama.cpp backend.
        )
        load_time = time.time() - load_start
        gpu_status = f"{n_gpu_layers} layers on GPU" if n_gpu_layers > 0 else "CPU mode"
        print(f"✅ Model loaded in {load_time:.2f} seconds ({gpu_status}, {n_threads} threads)")

        # --- GPIO Setup ---
        # This dictionary defines all the devices the assistant can control.
        # 'pin': The GPIO pin number using BOARD numbering.
        # 'state': The current state of the device (True=ON, False=OFF).
        # 'aliases': A list of alternative names for the device to improve recognition.
        self.gpio_devices = {
            'living_room_light': {'pin': 7, 'state': False, 'aliases': ['living room', 'main light']},
            'bedroom_light':     {'pin': 11, 'state': False, 'aliases': ['bedroom']},
            'kitchen_light':     {'pin': 13, 'state': False, 'aliases': ['kitchen']},
            'fan':               {'pin': 16, 'state': False, 'aliases': ['fan', 'air']},
            'heater':            {'pin': 18, 'state': False, 'aliases': ['heater', 'heat']},
        }
        self._setup_gpio()

        # --- Context & State ---
        # This dictionary holds contextual information that can be used by the assistant.
        self.context = {'temperature': 70}
        # Pre-defined patterns for the fast keyword matcher.
        self.patterns = {
            'cold': ['cold', 'freezing', 'chilly', 'cool'],
            'hot': ['hot', 'warm', 'sweating', 'boiling'],
            'dark': ['dark', 'dim', "can't see"],
            'bright': ['bright', 'blinding', 'too much light'],
        }

    def _setup_performance_tracking(self):
        """Initializes deques for tracking performance metrics."""
        # Deques are used for efficient appending and popping from both ends.
        # Here, they act as a sliding window of the last 100 measurements.
        self.performance_stats = {
            'token_speeds': deque(maxlen=100),
            'inference_latencies': deque(maxlen=100),
            'total_inferences': 0
        }

    def _setup_gpio(self):
        """Configures GPIO pins for all registered devices."""
        try:
            # This block attempts to initialize the real GPIO library.
            GPIO.setmode(GPIO.BOARD) # Use physical pin numbering.
            GPIO.setwarnings(False)  # Disable warnings about channels being already in use.
            for device in self.gpio_devices.values():
                # Set each pin as an output and initialize it to a low state (OFF).
                GPIO.setup(device['pin'], GPIO.OUT, initial=GPIO.LOW)
            print("✅ GPIO pins initialized.")
        except Exception as e:
            # If GPIO initialization fails (e.g., not on a Jetson/Pi),
            # it prints a warning and switches to a simulated GPIO control function.
            print(f"⚠️  GPIO Warning: {e}. Running in simulation mode.")
            self.control_gpio = self._control_gpio_simulated

    def _build_system_prompt(self):
        """
        Builds a detailed system prompt to guide the LLM into providing a structured
        JSON response. This is crucial for reliability.
        """
        # Get the current state of all devices to provide context to the LLM.
        device_states = {name: ('ON' if conf['state'] else 'OFF') for name, conf in self.gpio_devices.items()}
        
        # The system prompt is a set of instructions for the LLM. It defines its role,
        # the context, the available actions (intents), and the required output format.
        # Providing clear examples (few-shot prompting) drastically improves accuracy.
        return f"""You are a helpful smart home assistant. Your goal is to interpret user requests and translate them into a specific JSON format.

Current Context:
- Time of day: {self.get_time_context()}
- Temperature: {self.context['temperature']}°F
- Device States: {json.dumps(device_states)}

Available Intents:
- "turn_on": To turn specific devices on.
- "turn_off": To turn specific devices off.
- "increase_temp": To make the room warmer (turn heater on, fan off).
- "decrease_temp": To make the room cooler (turn fan on, heater off).
- "all_off": To turn all devices off.
- "status_check": To report the status of devices.
- "unknown": If the intent cannot be determined.

Based on the user's request, you must respond with ONLY a single JSON object. Do not add any explanations or conversational text outside of the JSON.

JSON format:
{{
  "intent": "intent_name",
  "devices": ["device_name_1", "device_name_2"],
  "reasoning": "A brief explanation of your decision.",
  "confidence": <a float between 0.0 and 1.0>
}}

Example 1:
User: "It's really dark in here."
JSON:
{{
  "intent": "turn_on",
  "devices": ["living_room_light", "kitchen_light"],
  "reasoning": "The user mentioned it's dark, implying a need for light. The living room and kitchen are common areas to light up.",
  "confidence": 0.85
}}

Example 2:
User: "I'm heading out, kill the power."
JSON:
{{
  "intent": "all_off",
  "devices": {list(self.gpio_devices.keys())},
  "reasoning": "The user is leaving and used the phrase 'kill the power', which means turn everything off.",
  "confidence": 0.95
}}

Example 3:
User: "What's on right now?"
JSON:
{{
  "intent": "status_check",
  "devices": [],
  "reasoning": "The user is asking for the current status of the devices.",
  "confidence": 1.0
}}
"""

    def _extract_json_from_response(self, text):
        """
        Extracts a JSON object from the model's raw text output.
        Handles cases where the JSON is embedded in markdown code blocks.
        """
        # The model might wrap the JSON in markdown code blocks (```json ... ```).
        # This regex looks for that pattern first.
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            json_str = match.group(1)
            print("Found JSON in markdown block.")
            return json.loads(json_str)
        
        # If not in a block, find the first string that looks like a JSON object.
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            try:
                json_str = match.group(0)
                return json.loads(json_str)
            except json.JSONDecodeError:
                print("⚠️  Found a JSON-like string but failed to parse.")
                return None
        return None

    def understand_command(self, user_input):
        """
        Processes user input, first with fast keyword matching, then with the LLM.
        This tiered approach provides speed for simple commands and intelligence for complex ones.
        """
        # Tier 1: Try a fast, explicit keyword match first.
        fast_result = self.fast_keyword_match(user_input)
        if fast_result['confidence'] >= 0.9:
            print(f"⚡ Fast match: {fast_result['intent']} (confidence: {fast_result['confidence']:.2f})")
            return fast_result

        # Tier 2: If no strong match, consult the LLM for deeper understanding.
        print("🧠 No fast match found. Consulting LLM...")
        inference_start = time.time()
        
        # Construct the full prompt including the system instructions and the user's query.
        prompt = f"{self._build_system_prompt()}\nUser: \"{user_input}\"\nJSON:\n"
        
        # Call the LLM for inference.
        response = self.llm(
            prompt,
            max_tokens=256,       # Limit the response length to prevent runaway generation.
            temperature=0.4,      # A lower temperature makes the output more deterministic and less "creative".
            top_p=0.9,            # Nucleus sampling parameter.
            repeat_penalty=1.15,  # Penalize the model for repeating itself.
            stop=["\nUser:", "}", "```"], # Tokens that will stop the generation immediately.
            echo=False            # Do not echo the prompt in the response.
        )
        
        generation_time = time.time() - inference_start
        # Sometimes the model stops at "}", so we add it back to ensure valid JSON.
        response_text = response['choices'][0]['text'].strip() + "}"

        # Update performance statistics.
        tokens_generated = response['usage']['completion_tokens']
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        self.performance_stats['token_speeds'].append(tokens_per_second)
        self.performance_stats['inference_latencies'].append(generation_time)
        self.performance_stats['total_inferences'] += 1
        print(f"⏱️  Inference: {generation_time*1000:.0f}ms | {tokens_per_second:.1f} tok/s")

        # Tier 3: Parse the LLM's response. If it fails, fall back to a keyword match.
        try:
            action = self._extract_json_from_response(response_text)
            if action:
                return action
            else:
                # Raise an error if no valid JSON could be extracted.
                raise json.JSONDecodeError("No valid JSON found", response_text, 0)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"❌ LLM Parse Error: {e}. Raw output:\n---\n{response_text}\n---")
            print("Falling back to simple interpretation.")
            return self.fast_keyword_match(user_input, is_fallback=True)

    def fast_keyword_match(self, user_input, is_fallback=False):
        """A simple, fast keyword-based interpreter for common commands."""
        text = user_input.lower()
        # If this is a fallback after an LLM failure, the confidence is lower.
        confidence = 0.95 if not is_fallback else 0.4
        
        # Check for explicit on/off commands.
        turn_on = 'turn on' in text or 'switch on' in text
        turn_off = 'turn off' in text or 'switch off' in text or 'kill' in text

        if turn_on or turn_off:
            intent = "turn_on" if turn_on else "turn_off"
            devices_to_action = []
            
            # Identify which specific devices were mentioned.
            for name, config in self.gpio_devices.items():
                if name.replace('_', ' ') in text or any(alias in text for alias in config['aliases']):
                    devices_to_action.append(name)
            
            # Handle general terms like "all" or "the lights".
            if 'all' in text or 'everything' in text:
                devices_to_action = list(self.gpio_devices.keys())
                if intent == "turn_off":
                    intent = "all_off" # Use a more specific intent for this case.
            elif 'the lights' in text and not devices_to_action:
                devices_to_action = [d for d in self.gpio_devices if 'light' in d]
            
            if devices_to_action:
                # Return a valid action object if devices were identified.
                return {"intent": intent, "devices": list(set(devices_to_action)), "confidence": confidence, "reasoning": "Direct keyword match."}

        # Handle contextual complaints (e.g., "I'm cold").
        if any(word in text for word in self.patterns['cold']):
            return {"intent": "increase_temp", "devices": ["heater"], "confidence": 0.85, "reasoning": "User complained about being cold."}
        if any(word in text for word in self.patterns['hot']):
            return {"intent": "decrease_temp", "devices": ["fan"], "confidence": 0.85, "reasoning": "User complained about being hot."}
        if any(word in text for word in self.patterns['dark']):
            return {"intent": "turn_on", "devices": ["living_room_light"], "confidence": 0.8, "reasoning": "User complained about darkness."}

        # If no keywords match, return an "unknown" intent.
        return {"intent": "unknown", "devices": [], "confidence": 0.1, "reasoning": "No keywords matched."}

    def control_gpio(self, device, action):
        """Controls a single GPIO device."""
        if device not in self.gpio_devices:
            print(f"Warning: Device '{device}' not found.")
            return False
        
        pin = self.gpio_devices[device]['pin']
        is_on = (action == 'on')
        
        # Set the GPIO pin to HIGH (on) or LOW (off).
        GPIO.output(pin, GPIO.HIGH if is_on else GPIO.LOW)
        # Update the internal state of the device.
        self.gpio_devices[device]['state'] = is_on
        return True

    def _control_gpio_simulated(self, device, action):
        """Simulated GPIO control for testing on non-Jetson/Pi systems."""
        if device not in self.gpio_devices:
            print(f"Warning: Device '{device}' not found.")
            return False
        is_on = (action == 'on')
        # Just update the internal state without touching any hardware.
        self.gpio_devices[device]['state'] = is_on
        print(f"[SIMULATED] Set {device} to {'ON' if is_on else 'OFF'}")
        return True

    def execute_command(self, action):
        """Executes the action determined by the understanding phase."""
        # Do not execute if the confidence is too low.
        if not action or action.get('confidence', 0) < 0.5:
            return "I'm not quite sure what you mean. Could you please rephrase that?"
        
        intent = action.get('intent', 'unknown')
        devices = action.get('devices', [])
        
        responses = []
        # Logic for turning devices on or off.
        if intent in ['turn_on', 'turn_off']:
            action_str = 'on' if intent == 'turn_on' else 'off'
            success = [d.replace('_', ' ') for d in devices if self.control_gpio(d, action_str)]
            if success:
                responses.append(f"Turned {action_str}: {', '.join(success)}.")
            else:
                responses.append("I couldn't find the specific device you mentioned.")
        
        # Logic for temperature control.
        elif intent == 'increase_temp':
            self.control_gpio('heater', 'on')
            self.control_gpio('fan', 'off')
            responses.append("Heater on, fan off. It should get warmer soon.")
        
        elif intent == 'decrease_temp':
            self.control_gpio('fan', 'on')
            self.control_gpio('heater', 'off')
            responses.append("Fan on, heater off. It should cool down shortly.")
            
        # Logic for turning everything off.
        elif intent == 'all_off':
            for device in self.gpio_devices:
                self.control_gpio(device, 'off')
            responses.append("Okay, I've turned everything off.")
            
        # Logic for checking device status.
        elif intent == 'status_check':
            on_devices = [name.replace('_', ' ') for name, conf in self.gpio_devices.items() if conf['state']]
            if on_devices:
                responses.append(f"Currently on: {', '.join(on_devices)}.")
            else:
                responses.append("Everything is currently off.")
        
        else:
            return f"I understood the intent '{intent}' but I can't perform that action."
            
        return " ".join(responses) if responses else "I'm sorry, I couldn't complete that request."

    def get_time_context(self):
        """Returns the current part of the day as a string."""
        hour = datetime.now().hour
        if 5 <= hour < 12: return "morning"
        if 12 <= hour < 17: return "afternoon"
        if 17 <= hour < 21: return "evening"
        return "night"

    def get_performance_summary(self):
        """Returns a formatted string of performance statistics."""
        if not self.performance_stats['inference_latencies']:
            return "No performance data yet."
        
        avg_latency = statistics.mean(self.performance_stats['inference_latencies']) * 1000
        avg_speed = statistics.mean(self.performance_stats['token_speeds'])
        
        return (f"📊 Performance:\n"
                f"├─ Avg Response: {avg_latency:.0f}ms\n"
                f"├─ Avg Speed: {avg_speed:.1f} tokens/sec\n"
                f"└─ Total LLM Inferences: {self.performance_stats['total_inferences']}")

    def cleanup(self):
        """Cleans up GPIO resources on exit to prevent issues."""
        print("\nShutting down and cleaning up GPIO...")
        try:
            GPIO.cleanup()
            print("✅ GPIO cleanup complete.")
        except Exception as e:
            print(f"⚠️  GPIO cleanup failed: {e}")

# --- Flask API ---
# This section sets up a simple web server to expose the assistant's functionality.
app = Flask(__name__)
assistant = None # Global assistant instance

@app.route('/')
def index():
    """A simple landing page for the API."""
    return """
    <h1>Smart Home API</h1>
    <p>POST /command with JSON `{"text": "your command"}`</p>
    <p>GET /status for device states and performance</p>
    """

@app.route('/command', methods=['POST'])
def process_command_api():
    """API endpoint to process a command."""
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'Request must be JSON with a "text" field.'}), 400
    
    user_input = data['text']
    action = assistant.understand_command(user_input)
    response_text = assistant.execute_command(action)
    
    return jsonify({
        'response': response_text,
        'action_taken': action,
        'device_states': {d: c['state'] for d, c in assistant.gpio_devices.items()}
    })

@app.route('/status', methods=['GET'])
def get_status():
    """API endpoint to get the current status of all devices and performance."""
    return jsonify({
        'device_states': {d: c['state'] for d, c in assistant.gpio_devices.items()},
        'performance': assistant.get_performance_summary(),
        'context': assistant.context
    })

def run_flask():
    """Runs the Flask web server."""
    # Use 'werkzeug' server for a more production-like environment if available.
    try:
        from werkzeug.serving import run_simple
        run_simple('0.0.0.0', 5000, app)
    except ImportError:
        app.run(host='0.0.0.0', port=5000)

# --- Main Execution Block ---
if __name__ == "__main__":
    # Set up an argument parser to handle command-line options.
    parser = argparse.ArgumentParser(description="Run the Smart Home Assistant.")
    parser.add_argument(
        "--model",
        type=str,
        default="/home/makatia/models/deepseek-7b.gguf",
        help="Path to the GGUF model file."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of CPU threads for the LLM."
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=35,
        help="Number of layers to offload to the GPU. Set to 0 for CPU-only."
    )
    args = parser.parse_args()

    try:
        # Initialize the main assistant class with the provided arguments.
        assistant = SmartHomeAssistant(
            model_path=args.model, 
            n_threads=args.threads, 
            n_gpu_layers=args.gpu_layers
        )
        
        # Start the Flask API in a separate, non-blocking thread.
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        
        print("\n\n🏠 Smart Home Assistant is Ready!")
        print("🗣️  Talk to it below, or use the API at http://<your_ip>:5000")
        print("ℹ️  Type 'perf' for stats, 'status' for device states, or 'quit' to exit.\n")
        
        # Main loop for command-line interaction.
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            if user_input.lower() == 'perf':
                print(assistant.get_performance_summary())
                continue
            if user_input.lower() == 'status':
                print(json.dumps(get_status().get_json(), indent=2))
                continue

            start_time = time.time()
            action = assistant.understand_command(user_input)
            response = assistant.execute_command(action)
            total_time = (time.time() - start_time) * 1000
            
            print(f"\nAssistant: {response}")
            # Print the LLM's reasoning for transparency.
            if action.get('reasoning'):
                print(f"Logic: {action['reasoning']} (Confidence: {action.get('confidence', 0):.2f})")
            print(f"[{total_time:.0f}ms]\n")

    except FileNotFoundError:
        print(f"❌ ERROR: Model file not found at '{args.model}'.")
        print("Please check the path and use the --model argument if necessary.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Ensure GPIO resources are cleaned up properly on exit.
        if assistant:
            assistant.cleanup()
        print("Shutdown complete.")
