# Regional Adaptive Sampling

[Regional Adaptive Sampling](https://github.com/microsoft/RAS) is a new technique for accelerating the inference of diffusion transformers. 
It essentially works as a KV Cache inside the model, picking regions that are likely to be updated by each diffusion step and passing in only those tokens.

This implementation is simple to use, and compatible with Flux (dev & schnell) at HunYuanVideo. I may add support for other models in the future.

## Usage
Apply the `Regional Adaptive Sampling` node to the desired model. It has the following parameters:  
- **sample_ratio**: The percent of tokens to keep in the model on a RAS pass. Anything below 0.3 is usually very bad quality.
- **warmup_steps**: The number of steps to do without RAS at the beginning. Setting higher will decrease the speedup, and setting it lower will degrade the composition.
- **hydrate_every**: Every `hydrate_every` steps, we do a full run through the model with all tokens, to refresh the stale cache. Set to 0 to disable and do full RAS.
- **starvation_scale**: Controls how the model decides which part of the image to focus on. Increasing it will probably shift quality from the main subject to the background. The default of 0.1 is what's used in the paper, and I haven't tried anything else.
