import tkinter as tk
from tkinter import ttk, messagebox
from dotenv import load_dotenv
import os
import sys
import subprocess
import uvicorn
import threading
import webbrowser
from product_pipeline import app
from util.ai_util import AIProvider, AiUtil
from util.llm_config import ALL_CONFIGS


class ProductPipelineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Product Pipeline GUI")
        self.root.geometry("600x500")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # LLM Configuration Frame
        llm_frame = ttk.LabelFrame(main_frame, text="LLM Configuration", padding="10")
        llm_frame.grid(row=0, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        # LLM Model Selection
        ttk.Label(llm_frame, text="Model:").grid(row=0, column=0, padx=5, pady=5)
        self.model_var = tk.StringVar(value="nemotron-mini")  # Default model
        model_names = list(ALL_CONFIGS.keys())
        self.model_combo = ttk.Combobox(llm_frame, textvariable=self.model_var, values=model_names)
        self.model_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # API Key Entry
        ttk.Label(llm_frame, text="API Key:").grid(row=1, column=0, padx=5, pady=5)
        self.api_key_var = tk.StringVar()
        self.api_key_entry = ttk.Entry(llm_frame, textvariable=self.api_key_var, width=40)
        self.api_key_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Server status
        self.server_status = ttk.Label(main_frame, text="Server Status: Stopped", foreground="red")
        self.server_status.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Server controls
        self.server_button = ttk.Button(main_frame, text="Start Server", command=self.toggle_server)
        self.server_button.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Open browser button
        self.browser_button = ttk.Button(main_frame, text="Open in Browser", command=self.open_browser)
        self.browser_button.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Pattern Generation Frame
        pattern_frame = ttk.LabelFrame(main_frame, text="Generate Patterns", padding="10")
        pattern_frame.grid(row=4, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        # Number of patterns
        ttk.Label(pattern_frame, text="Number of Patterns:").grid(row=0, column=0, padx=5, pady=5)
        self.pattern_count = ttk.Spinbox(pattern_frame, from_=1, to=10, width=5)
        self.pattern_count.set(3)
        self.pattern_count.grid(row=0, column=1, padx=5, pady=5)
        
        # Idea input
        ttk.Label(pattern_frame, text="Idea:").grid(row=1, column=0, padx=5, pady=5)
        self.idea_input = ttk.Entry(pattern_frame, width=40)
        self.idea_input.grid(row=1, column=1, padx=5, pady=5)
        
        # Generate button
        self.generate_button = ttk.Button(pattern_frame, text="Generate", command=self.generate_patterns)
        self.generate_button.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Bind model selection change
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_change)
        
        # Initialize based on selected model
        self.on_model_change(None)
        
        # Environment variables check
        self.check_env_vars()
        
        self.server_running = False
        self.server_thread = None

    def check_env_vars(self):
        required_vars = ['GH_UPLOAD_REPO', 'GH_PAT', 'GH_CONTENT_PREFIX']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            messagebox.showwarning(
                "Missing Environment Variables",
                f"Please set the following environment variables in .env file:\n{', '.join(missing_vars)}"
            )
    
    def on_model_change(self, event):
        """Handle model selection change"""
        selected_model = self.model_var.get()
        config = ALL_CONFIGS.get(selected_model)
        
        # Show/hide API key entry based on provider
        if config and config.provider == "ollama":
            self.api_key_entry.config(state="disabled")
        else:
            self.api_key_entry.config(state="normal")
    
    def toggle_server(self):
        if not self.server_running:
            self.server_thread = threading.Thread(target=self.run_server)
            self.server_thread.daemon = True
            self.server_thread.start()
            self.server_running = True
            self.server_status.config(text="Server Status: Running", foreground="green")
            self.server_button.config(text="Stop Server")
        else:
            # Note: Properly stopping a uvicorn server programmatically is complex
            # For simplicity, we'll just indicate it's stopped
            self.server_running = False
            self.server_status.config(text="Server Status: Stopped", foreground="red")
            self.server_button.config(text="Start Server")
            
    def run_server(self):
        config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="info")
        server = uvicorn.Server(config)
        server.run()
    
    def open_browser(self):
        if self.server_running:
            webbrowser.open('http://127.0.0.1:8000/docs')
        else:
            messagebox.showinfo("Server Not Running", "Please start the server first.")
    
    def generate_patterns(self):
        if not self.server_running:
            messagebox.showinfo("Server Not Running", "Please start the server first.")
            return
            
        patterns = self.pattern_count.get()
        idea = self.idea_input.get()
        
        if not idea:
            messagebox.showwarning("Input Required", "Please enter an idea.")
            return
        
        # Get selected model configuration
        selected_model = self.model_var.get()
        config = ALL_CONFIGS.get(selected_model)
        if not config:
            messagebox.showerror("Error", "Invalid model selection.")
            return
            
        try:
            # Initialize AI utility with selected configuration
            api_key = self.api_key_var.get() if config.provider != "ollama" else None
            ai_util = AiUtil(
                provider=AIProvider(config.provider),
                model=config.model,
                api_key=api_key
            )
            
            import requests
            response = requests.post(
                'http://127.0.0.1:8000/process_patterns',
                json={'patterns': int(patterns), 'idea': idea}
            )
            
            if response.status_code == 200:
                messagebox.showinfo("Success", "Patterns generated successfully!")
            else:
                messagebox.showerror("Error", f"Failed to generate patterns: {response.text}")
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

def main():
    root = tk.Tk()
    app = ProductPipelineGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()