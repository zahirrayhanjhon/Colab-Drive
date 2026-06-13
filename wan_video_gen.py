import os
import time
from pathlib import Path

def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

def mount_drive():
    if is_colab():
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted successfully.")
    else:
        print("⚠️ Not running in Google Colab. Skipping Drive mount.")

class WanVideoGenerator:
    """
    Helper class to run Wan 2.2 Video generation models on Google Colab (T4 compatible).
    Caches models and saves outputs to a specific Google Drive folder to persist data.
    """
    def __init__(self, drive_folder_path="/content/drive/MyDrive/WanVideoGen"):
        # Auto-install dependencies if not found
        if is_colab():
            try:
                import diffusers
                import accelerate
            except ImportError:
                print("📦 Required packages not found. Installing now...")
                os.system("pip install -q -U diffusers transformers accelerate imageio[ffmpeg] sentencepiece protobuf")
                print("✅ Packages installed.")

        self.base_dir = Path(drive_folder_path) if is_colab() else Path("./WanVideoGen")
        self.models_dir = self.base_dir / "models"
        self.outputs_dir = self.base_dir / "outputs"
        self.inputs_dir = self.base_dir / "inputs"
        
        if is_colab():
            mount_drive()
            
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        
        # Set HuggingFace cache to Drive so we don't redownload models every session
        os.environ["HF_HOME"] = str(self.models_dir)
        
        self.t2v_pipe = None
        self.i2v_pipe = None
        
        print(f"📁 Workspace initialized at: {self.base_dir}")
        print("🚀 Ready to generate videos!")

    def _load_t2v_pipeline(self):
        if self.t2v_pipe is None:
            import torch
            from diffusers import WanPipeline
            
            print(f"⏳ Loading Text-to-Video model (Wan2.2-T2V-A14B)...")
            print(f"📥 Checking Drive cache at {self.models_dir}... (Will download if empty)")
            print("⚠️ WARNING: This is a 14B MoE model. CPU offloading will be enabled for T4 compatibility.")
            model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
            self.t2v_pipe = WanPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.bfloat16,
                cache_dir=str(self.models_dir)  # Explicitly save to Drive
            )
            # Memory optimizations for T4 GPU
            self.t2v_pipe.enable_model_cpu_offload()
            self.t2v_pipe.vae.enable_tiling()
            
            print("✅ Text-to-Video model loaded with CPU offloading and VAE tiling!")
        return self.t2v_pipe

    def _load_i2v_pipeline(self):
        if self.i2v_pipe is None:
            import torch
            from diffusers import WanImageToVideoPipeline
            
            print(f"⏳ Loading Image-to-Video model (Wan2.2-I2V-A14B)...")
            print(f"📥 Checking Drive cache at {self.models_dir}... (Will download if empty)")
            print("⚠️ WARNING: This is a 14B MoE model. CPU offloading will be enabled for T4 compatibility.")
            model_id = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
            self.i2v_pipe = WanImageToVideoPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.bfloat16,
                cache_dir=str(self.models_dir)  # Explicitly save to Drive
            )
            # Memory optimizations for T4 GPU to avoid Out-Of-Memory errors
            self.i2v_pipe.enable_model_cpu_offload()
            self.i2v_pipe.vae.enable_tiling()
            
            print("✅ Image-to-Video model loaded with CPU offloading and VAE tiling!")
        return self.i2v_pipe

    def generate_text_to_video(self, prompt, num_frames=81, fps=16, guidance_scale=5.0):
        """
        Generates a video from text using the 14B T2V model.
        Default generates ~5 seconds of video at 16 fps.
        """
        import torch
        from diffusers.utils import export_to_video
        
        pipe = self._load_t2v_pipeline()
        
        print(f"🎬 Generating video for prompt: '{prompt}'")
        output = pipe(
            prompt=prompt,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
        ).frames[0]
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"t2v_{timestamp}.mp4"
        output_path = str(self.outputs_dir / filename)
        
        export_to_video(output, output_path, fps=fps)
        print(f"🎉 Video saved to: {output_path}")
        
        self.display_video(output_path)
        return output_path

    def generate_image_to_video(self, image_path_or_url, prompt, num_frames=81, fps=16, guidance_scale=5.0):
        """
        Generates a video from an image and text using the 14B I2V model.
        """
        import torch
        from diffusers.utils import load_image, export_to_video
        
        pipe = self._load_i2v_pipeline()
        
        print(f"🖼️ Loading image from: {image_path_or_url}")
        image = load_image(image_path_or_url)
        
        print(f"🎬 Animating image with prompt: '{prompt}'")
        print("⏳ Note: Image-to-Video on T4 GPU takes approx 20-30 mins for 81 frames due to CPU offloading.")
        output = pipe(
            image=image,
            prompt=prompt,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
        ).frames[0]
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"i2v_{timestamp}.mp4"
        output_path = str(self.outputs_dir / filename)
        
        export_to_video(output, output_path, fps=fps)
        print(f"🎉 Video saved to: {output_path}")
        
        self.display_video(output_path)
        return output_path

    def display_video(self, video_path):
        """Displays a generated video inside the Colab notebook."""
        if not is_colab():
            return
            
        from IPython.display import HTML, display
        from base64 import b64encode
        
        print(f"📺 Displaying: {video_path}")
        try:
            mp4 = open(video_path, 'rb').read()
            data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
            
            display(HTML(f'''
            <video width="480" controls autoplay loop>
                <source src="{data_url}" type="video/mp4">
            </video>
            '''))
        except Exception as e:
            print(f"⚠️ Could not display video inline: {e}")

def setup_environment():
    """Helper to install necessary packages in Colab."""
    print("📦 Installing required packages...")
    os.system("pip install -q -U diffusers transformers accelerate imageio[ffmpeg] sentencepiece protobuf")
    print("✅ Packages installed. You may need to restart the session if this is the first install.")

if __name__ == "__main__":
    print("💡 Import WanVideoGenerator to use:")
    print("   from wan_video_gen import WanVideoGenerator")
    print("   gen = WanVideoGenerator()")
    print("   gen.generate_text_to_video('A cute cat.')")
