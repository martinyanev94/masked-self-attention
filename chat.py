import torch


def chat(model, tokenizer, max_new_tokens=200):
    print("Chat with your GPT model! Type 'exit' to quit.")
    device = next(model.parameters()).device

    history = ""  # conversation log

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Append user message to history
        history += f"User: {user_input}\nAI: "

        # Encode
        idx = tokenizer.encode(history)
        idx = torch.tensor([idx], dtype=torch.long, device=device)

        # Generate model response
        out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=30)

        # Decode only the NEW tokens
        decoded = tokenizer.decode(out[0].tolist())
        ai_response = decoded[len(history):].strip()

        print(f"AI: {ai_response}")

        # Add back to history so model keeps context
        history += ai_response + "\n"