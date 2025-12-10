#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive Web Demo using Gradio
3-Model Inference Comparison Interface with Real-time Performance Monitoring
"""

import gradio as gr
import torch
import time
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import psutil

from mistral_fp16 import MistralFP16Loader, load_quantized_model
from utils import setup_logging

logger = setup_logging('logs/app.log')

class GradioDemo:
    """
    Gradio web interface for model inference comparison
    """
    
    def __init__(self):
        """
        Initialize demo
        """
        self.models = {}
        self.tokenizers = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.chat_history = []
        self.history_file = Path('outputs/chat_history.csv')
        
        # Initialize chat history file
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.history_file.exists():
            with open(self.history_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Prompt', 'Model', 'Response', 'Time(s)', 'Memory(GB)', 'Tokens/s'])
        
        logger.info("Gradio Demo initialized")
        self.load_models()
    
    def load_models(self):
        """
        Load all models
        """
        print("\n🔄 Loading models for demo...")
        
        # Load FP16
        try:
            fp16_loader = MistralFP16Loader()
            self.models['mistral_fp16'], self.tokenizers['mistral_fp16'] = fp16_loader.load_model()
            print("✅ Mistral FP16 loaded")
        except Exception as e:
            print(f"❌ Failed to load Mistral FP16: {e}")
            logger.error(f"Failed to load Mistral FP16: {e}")
        
        # Load 4-bit
        try:
            if Path('models/mistral-7b-4bit').exists():
                self.models['mistral_4bit'], self.tokenizers['mistral_4bit'] = load_quantized_model(
                    'models/mistral-7b-4bit'
                )
                print("✅ Mistral 4-bit loaded")
            else:
                print("⚠️  Mistral 4-bit model not found")
        except Exception as e:
            print(f"❌ Failed to load Mistral 4-bit: {e}")
            logger.error(f"Failed to load Mistral 4-bit: {e}")
        
        # Load Distilled Phi-2
        try:
            if Path('models/phi-2-distilled').exists():
                from transformers import AutoTokenizer, AutoModelForCausalLM
                self.tokenizers['phi2_distilled'] = AutoTokenizer.from_pretrained(
                    'models/phi-2-distilled'
                )
                self.models['phi2_distilled'] = AutoModelForCausalLM.from_pretrained(
                    'models/phi-2-distilled',
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                print("✅ Phi-2 Distilled loaded")
            else:
                print("⚠️  Phi-2 Distilled model not found")
        except Exception as e:
            print(f"❌ Failed to load Phi-2 Distilled: {e}")
            logger.error(f"Failed to load Phi-2 Distilled: {e}")
        
        print(f"\n✅ Loaded {len(self.models)}/3 models")
    
    def get_system_memory(self) -> float:
        """
        Get current system memory usage in GB
        """
        if self.device == 'cuda':
            return torch.cuda.memory_allocated() / (1024**3)
        else:
            process = psutil.Process()
            return process.memory_info().rss / (1024**3)
    
    def run_inference(self, prompt: str, model_name: str, max_length: int = 100) -> Tuple[str, float, float, float]:
        """
        Run inference for a single model
        
        Returns: (response, time_taken, memory_used, tokens_per_sec)
        """
        if model_name not in self.models:
            return f"Error: Model {model_name} not loaded", 0, 0, 0
        
        try:
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]
            device = next(model.parameters()).device
            
            # Record initial memory
            initial_memory = self.get_system_memory()
            
            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(device)
            
            # Inference
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            elapsed_time = time.time() - start_time
            
            # Memory usage
            final_memory = self.get_system_memory()
            memory_used = final_memory - initial_memory
            
            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            tokens_generated = outputs.shape[1]
            tokens_per_sec = tokens_generated / elapsed_time if elapsed_time > 0 else 0
            
            return response, elapsed_time, memory_used, tokens_per_sec
            
        except Exception as e:
            logger.error(f"Inference error in {model_name}: {e}")
            return f"Error: {str(e)}", 0, 0, 0
    
    def process_query(self, prompt: str) -> Tuple[str, str, str, str, str, str]:
        """
        Process user query with all models
        """
        if not prompt.strip():
            return "", "", "", "", "", "Please enter a prompt!"
        
        # Run inference for each model
        results = {}
        for model_name in ['mistral_fp16', 'mistral_4bit', 'phi2_distilled']:
            if model_name in self.models:
                response, time_taken, memory_used, tokens_per_sec = self.run_inference(
                    prompt, model_name, max_length=100
                )
                results[model_name] = {
                    'response': response,
                    'time': time_taken,
                    'memory': memory_used,
                    'tokens_per_sec': tokens_per_sec
                }
                
                # Save to history
                self.save_to_history(prompt, model_name, response, time_taken, memory_used, tokens_per_sec)
        
        # Format outputs
        fp16_output = self._format_output(results.get('mistral_fp16', {}))
        bit4_output = self._format_output(results.get('mistral_4bit', {}))
        distilled_output = self._format_output(results.get('phi2_distilled', {}))
        
        # Performance table
        perf_table = self._create_performance_table(results)
        
        return fp16_output, bit4_output, distilled_output, perf_table, "", "Processing complete!"
    
    def _format_output(self, result: dict) -> str:
        """
        Format output for display
        """
        if not result:
            return "[Model not loaded]"
        
        if 'error' in result or 'response' not in result:
            return f"[Error]"
        
        output = f"{result['response'][:500]}"
        if len(result['response']) > 500:
            output += "..."
        
        output += f"\n\n⏱️ Time: {result['time']:.3f}s"
        output += f" | 💾 Memory: {result['memory']:.2f}GB"
        output += f" | 💫 {result['tokens_per_sec']:.2f}t/s"
        
        return output
    
    def _create_performance_table(self, results: dict) -> str:
        """
        Create performance comparison table
        """
        table = "### Performance Comparison\n\n"
        table += "| Model | Time (s) | Memory (GB) | Tokens/s |\n"
        table += "|-------|----------|-------------|----------|\n"
        
        for model_name, result in results.items():
            if result and 'response' in result:
                display_name = model_name.replace('_', '-').title()
                table += f"| {display_name} | {result['time']:.3f} | {result['memory']:.2f} | {result['tokens_per_sec']:.2f} |\n"
        
        return table
    
    def save_to_history(self, prompt: str, model: str, response: str, time_taken: float, memory: float, tokens_per_sec: float):
        """
        Save inference to chat history CSV
        """
        try:
            with open(self.history_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    prompt[:100],  # Truncate long prompts
                    model,
                    response[:100],  # Truncate long responses
                    f"{time_taken:.3f}",
                    f"{memory:.2f}",
                    f"{tokens_per_sec:.2f}"
                ])
        except Exception as e:
            logger.error(f"Failed to save to history: {e}")

def create_interface() -> gr.Blocks:
    """
    Create Gradio interface
    """
    demo = GradioDemo()
    
    with gr.Blocks(title="🤖 Mistral LLM Comparison Demo") as interface:
        # Header
        gr.Markdown("# 🤖 Mistral LLM Model Comparison")
        gr.Markdown("### Compare FP16, 4-bit Quantization, and Distilled Models")
        gr.Markdown("""  
This demo compares inference performance across three model configurations:
- **Mistral FP16**: Full precision (16GB VRAM)
- **Mistral 4-bit**: 4-bit quantized (4GB VRAM)
- **Phi-2 Distilled**: Knowledge distilled version (2GB VRAM)
""")
        
        # Input section
        with gr.Row():
            with gr.Column(scale=4):
                prompt_input = gr.Textbox(
                    label="Enter your prompt",
                    placeholder="Type your question here...",
                    lines=3
                )
            with gr.Column(scale=1):
                submit_button = gr.Button(
                    value="Run Inference",
                    scale=1,
                    size="lg"
                )
        
        # Output section
        gr.Markdown("## Model Outputs")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Mistral FP16")
                mistral_fp16_output = gr.Textbox(
                    label="FP16 Output",
                    interactive=False,
                    lines=8
                )
            
            with gr.Column():
                gr.Markdown("### Mistral 4-bit")
                mistral_4bit_output = gr.Textbox(
                    label="4-bit Output",
                    interactive=False,
                    lines=8
                )
            
            with gr.Column():
                gr.Markdown("### Phi-2 Distilled")
                phi2_output = gr.Textbox(
                    label="Distilled Output",
                    interactive=False,
                    lines=8
                )
        
        # Performance table
        with gr.Row():
            performance_table = gr.Markdown()
        
        # Status
        with gr.Row():
            with gr.Column():
                prompt_display = gr.Textbox(
                    label="Last Prompt",
                    interactive=False
                )
            with gr.Column():
                status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Ready"
                )
        
        # Connect button
        submit_button.click(
            fn=demo.process_query,
            inputs=[prompt_input],
            outputs=[
                mistral_fp16_output,
                mistral_4bit_output,
                phi2_output,
                performance_table,
                prompt_display,
                status
            ]
        )
        
        # Footer
        gr.Markdown("""
---
### Information
- **FP16 Model**: Uses full 16-bit precision. Largest memory footprint but potentially highest quality.
- **4-bit Model**: Uses BitsAndBytes 4-bit quantization. Reduces memory by 75% with minimal quality loss.
- **Distilled Model**: Smaller student model trained via knowledge distillation. Smallest footprint.

### Quick Links
- Source Code: [GitHub](https://github.com/caizongxun/mistral-quantization-distillation)
- Mistral Docs: [mistralai.com](https://www.mistralai.com/)
""")
    
    return interface

if __name__ == '__main__':
    interface = create_interface()
    interface.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        debug=False
    )
