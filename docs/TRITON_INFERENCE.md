# Triton Inference Server Guidelines for SynthGen

This document provides guidelines for deploying the SynthGen text-to-audio model using NVIDIA Triton Inference Server. Triton enables high-performance, scalable inference in production environments by supporting concurrent model execution, dynamic batching, and multiple backend frameworks.

## Deployment Architecture

Deploying SynthGen on Triton requires splitting the pipeline into two separate models to optimize performance and resource utilization:

1. **Text Encoder (T5-base)**: Handled by an ONNX or TensorRT backend.
2. **Audio Generation (DiT + VAE)**: Handled by a Python backend, executing the Flow Matching loop and VAE decoding.

This decoupled architecture allows the text encoder to scale independently of the heavier diffusion generation process.

## 1. Exporting the Text Encoder

The T5-base text encoder should be exported to ONNX format for optimal performance within Triton.

### Export Script

```python
import torch
from transformers import T5EncoderModel, T5Tokenizer

model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5EncoderModel.from_pretrained(model_name)

# Dummy inputs
dummy_text = ["A warm analog synthesizer pad"]
inputs = tokenizer(dummy_text, return_tensors="pt", max_length=256, padding="max_length", truncation=True)

# Export to ONNX
torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"]),
    "t5_encoder.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["text_embeds"],
    dynamic_axes={
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "text_embeds": {0: "batch_size"}
    },
    opset_version=14
)
```

### Triton Model Repository Structure

Create the following directory structure for the text encoder:

```
model_repository/
└── text_encoder/
    ├── 1/
    │   └── model.onnx
    └── config.pbtxt
```

### Configuration (`config.pbtxt`)

```protobuf
name: "text_encoder"
platform: "onnxruntime_onnx"
max_batch_size: 16

input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ 256 ]
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ 256 ]
  }
]

output [
  {
    name: "text_embeds"
    data_type: TYPE_FP32
    dims: [ 256, 768 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

## 2. Deploying the Audio Generator (Python Backend)

The generation loop involves iterative sampling (Flow Matching) and cannot be easily exported to a static graph. Therefore, we use the Triton Python Backend to execute the DiT and VAE.

### Triton Model Repository Structure

```
model_repository/
└── audio_generator/
    ├── 1/
    │   └── model.py
    └── config.pbtxt
```

### Configuration (`config.pbtxt`)

```protobuf
name: "audio_generator"
backend: "python"
max_batch_size: 8

input [
  {
    name: "text_embeds"
    data_type: TYPE_FP32
    dims: [ 256, 768 ]
  },
  {
    name: "duration"
    data_type: TYPE_FP32
    dims: [ 1 ]
  },
  {
    name: "num_steps"
    data_type: TYPE_INT32
    dims: [ 1 ]
  },
  {
    name: "cfg_scale"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }
]

output [
  {
    name: "audio"
    data_type: TYPE_FP32
    dims: [ 2, -1 ]
  },
  {
    name: "sample_rate"
    data_type: TYPE_INT32
    dims: [ 1 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

### Python Backend Implementation (`model.py`)

The `model.py` script loads the SynthGen checkpoint and handles the inference loop.

```python
import json
import triton_python_backend_utils as pb_utils
import torch
import numpy as np
import sys
import os

# Add the repository to the Python path
sys.path.append("/workspace/synthgen-experimental")
from synthgen.model.synthgen import SynthGen

class TritonPythonModel:
    def initialize(self, args):
        """Initialize the model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the checkpoint (path should be mapped into the container)
        checkpoint_path = "/workspace/checkpoints/synthgen_final.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        config = checkpoint.get("config", {})
        self.model = SynthGen(
            vae_latent_dim=config.get("vae_latent_dim", 64),
            dit_model_dim=config.get("dit_model_dim", 1024),
            dit_num_heads=config.get("dit_num_heads", 16),
            dit_num_layers=config.get("dit_num_layers", 20),
            use_dummy_text_encoder=True  # Text encoding is handled externally
        )
        
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.model.to(self.device, dtype=torch.bfloat16)
        self.model.eval()

    def execute(self, requests):
        """Execute inference for a batch of requests."""
        responses = []
        
        for request in requests:
            # Extract inputs
            text_embeds = pb_utils.get_input_tensor_by_name(request, "text_embeds").as_numpy()
            duration = pb_utils.get_input_tensor_by_name(request, "duration").as_numpy()[0][0]
            num_steps = pb_utils.get_input_tensor_by_name(request, "num_steps").as_numpy()[0][0]
            cfg_scale = pb_utils.get_input_tensor_by_name(request, "cfg_scale").as_numpy()[0][0]
            
            # Convert to tensors
            text_embeds_pt = torch.from_numpy(text_embeds).to(self.device, dtype=torch.bfloat16)
            
            # Setup unconditional embeddings for CFG
            batch_size = text_embeds_pt.shape[0]
            text_embeds_uncond = torch.zeros_like(text_embeds_pt)
            dur_tensor = torch.full((batch_size,), duration, device=self.device, dtype=torch.float32)
            
            # Calculate latent length
            audio_samples = int(duration * self.model.sample_rate)
            latent_length = self.model.vae.get_latent_length(audio_samples)
            
            # Sample noise
            noise = torch.randn(
                batch_size, self.model.vae_latent_dim, latent_length,
                device=self.device, dtype=torch.bfloat16
            )
            
            # Generate
            with torch.no_grad():
                def model_fn(x_t, t, text_emb, dur):
                    return self.model.dit(x_t, t, text_emb, dur)
                
                latents = self.model.scheduler.sample(
                    model_fn=model_fn,
                    noise=noise,
                    num_steps=int(num_steps),
                    cfg_scale=float(cfg_scale),
                    text_embeds=text_embeds_pt,
                    text_embeds_uncond=text_embeds_uncond,
                    duration=dur_tensor,
                )
                
                audio = self.model.vae.decode(latents)
                audio = audio[..., :audio_samples].cpu().float().numpy()
            
            # Create response
            out_audio = pb_utils.Tensor("audio", audio)
            out_sr = pb_utils.Tensor("sample_rate", np.array([self.model.sample_rate], dtype=np.int32))
            
            response = pb_utils.InferenceResponse(output_tensors=[out_audio, out_sr])
            responses.append(response)
            
        return responses
```

## 3. Starting the Server

Launch the Triton Inference Server using Docker, ensuring the model repository and the codebase are mounted correctly.

```bash
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /path/to/model_repository:/models \
  -v /path/to/synthgen-experimental:/workspace/synthgen-experimental \
  -v /path/to/checkpoints:/workspace/checkpoints \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models
```

## 4. Client Integration

To query the server, you must orchestrate the calls: first to the `text_encoder`, then pass the resulting embeddings to the `audio_generator`.

```python
import numpy as np
import tritonclient.http as httpclient
from transformers import T5Tokenizer

client = httpclient.InferenceServerClient(url="localhost:8000")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

def generate_audio(prompt, duration=10.0):
    # 1. Text Encoding
    inputs = tokenizer([prompt], return_tensors="np", max_length=256, padding="max_length", truncation=True)
    
    text_inputs = [
        httpclient.InferInput("input_ids", inputs["input_ids"].shape, "INT64"),
        httpclient.InferInput("attention_mask", inputs["attention_mask"].shape, "INT64")
    ]
    text_inputs[0].set_data_from_numpy(inputs["input_ids"])
    text_inputs[1].set_data_from_numpy(inputs["attention_mask"])
    
    text_result = client.infer(model_name="text_encoder", inputs=text_inputs)
    text_embeds = text_result.as_numpy("text_embeds")
    
    # 2. Audio Generation
    audio_inputs = [
        httpclient.InferInput("text_embeds", text_embeds.shape, "FP32"),
        httpclient.InferInput("duration", [1, 1], "FP32"),
        httpclient.InferInput("num_steps", [1, 1], "INT32"),
        httpclient.InferInput("cfg_scale", [1, 1], "FP32")
    ]
    
    audio_inputs[0].set_data_from_numpy(text_embeds)
    audio_inputs[1].set_data_from_numpy(np.array([[duration]], dtype=np.float32))
    audio_inputs[2].set_data_from_numpy(np.array([[25]], dtype=np.int32))
    audio_inputs[3].set_data_from_numpy(np.array([[3.5]], dtype=np.float32))
    
    audio_result = client.infer(model_name="audio_generator", inputs=audio_inputs)
    
    audio_data = audio_result.as_numpy("audio")
    sample_rate = audio_result.as_numpy("sample_rate")[0]
    
    return audio_data, sample_rate
```
