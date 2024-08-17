def chunk_text(text, chunk_size=512):
    # Simple strategy: split by sentence, or more advanced: split by semantic coherence
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunkss