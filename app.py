import os
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re  

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

shipping_data = [
    {"weight": 2, "distance": 500, "carrier": "FedEx", "cost": 15},
    {"weight": 5, "distance": 200, "carrier": "UPS", "cost": 10},
    {"weight": 1, "distance": 800, "carrier": "DHL", "cost": 20},
    {"weight": 3, "distance": 700, "carrier": "USPS", "cost": 18},
    {"weight": 6, "distance": 300, "carrier": "Amazon Shipping", "cost": 12}
]

class ShippingOption(BaseModel):
    carrier: str
    estimated_cost: float

class AIResponse(BaseModel):
    response: str

model_name = "tiiuae/falcon-7b-instruct"  
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id 

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  
    device_map="auto", 
    trust_remote_code=False  
)

def generate_response(prompt: str, max_length: int = 200) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    outputs = model.generate(
        inputs["input_ids"],  
        attention_mask=inputs["attention_mask"],  
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.9,  
        temperature=0.7, 
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id  
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def find_best_shipping_option(weight: float, distance: float) -> ShippingOption:
    suitable_options = [option for option in shipping_data if option["weight"] >= weight]
    if not suitable_options:
        raise HTTPException(status_code=404, detail="No suitable shipping options found.")
    
    best_option = min(suitable_options, key=lambda x: abs(x["distance"] - distance))
    return ShippingOption(carrier=best_option["carrier"], estimated_cost=best_option["cost"])

@app.get("/predict-shipping", response_model=ShippingOption)
def predict_shipping(weight: float = Query(..., description="Weight of the package in kg"),
                     distance: float = Query(..., description="Distance in km")):
    return find_best_shipping_option(weight, distance)



@app.get("/ask-ai", response_model=AIResponse)
def ask_ai(query: str = Query(..., description="Your question about shipping")):
    try:
        weight_match = re.search(r"(\d+)\s*kg", query)  
        distance_match = re.search(r"(\d+)\s*km", query)  

        weight = float(weight_match.group(1)) if weight_match else 2  
        distance = float(distance_match.group(1)) if distance_match else 500  

        best_option = find_best_shipping_option(weight, distance)

        prompt = f"""Based on the following data:
        - Weight: {weight} kg
        - Distance: {distance} km
        - Best Carrier: {best_option.carrier}
        - Estimated Cost: ${best_option.estimated_cost}

        Answer the following question concisely: {query}"""
        
        ai_response = generate_response(prompt)

        cleaned_response = ai_response.split("\n")[-1].strip()
        return AIResponse(response=cleaned_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI request failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)