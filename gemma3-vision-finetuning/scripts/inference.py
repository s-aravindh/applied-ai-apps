from transformers import TextStreamer
from config import GenerationConfig

def generate_response(model, tokenizer, image=None, text="", 
                     max_new_tokens=GenerationConfig.MAX_NEW_TOKENS, stream=False):
    messages = [{
        "role": "user",
        "content": []
    }]
    
    if image is not None:
        messages[0]["content"].append({
            "type": "image",
            "image": image
        })
    
    messages[0]["content"].append({
        "type": "text",
        "text": text
    })

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    streamer = TextStreamer(tokenizer, skip_prompt=True) if stream else None
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=GenerationConfig.TEMPERATURE,
        top_p=GenerationConfig.TOP_P,
        top_k=GenerationConfig.TOP_K,
        streamer=streamer,
    )
    
    if not stream:
        return tokenizer.batch_decode(outputs)
    return None
