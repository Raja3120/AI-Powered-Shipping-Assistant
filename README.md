
# AI-Powered Shipping Assistant

## Overview
The **AI-Powered Shipping Assistant** is a project that combines custom shipping data with the **Falcon language model** to provide accurate and intelligent responses to shipping-related queries. It includes two main features:
1. **Shipping Cost Prediction**: Predicts the best shipping option based on weight and distance.
2. **AI Shipping Assistant**: Answers natural language questions about shipping using the Falcon model.

This project demonstrates how AI can simplify complex tasks like shipping logistics, making it easier for users to find the best shipping options quickly.

## Features
- **Shipping Cost Prediction**: Given weight and distance, the system predicts the best shipping option.
- **AI Shipping Assistant**: Answers natural language questions about shipping, combining the Falcon model with custom shipping data.
- **Dynamic Query Parsing**: Extracts weight and distance from user queries using regex.
- **REST API**: Built with FastAPI for easy integration with other applications.

## Installation

### Prerequisites
- Python 3.8 or higher.
- A machine with a GPU (recommended for faster inference).

### Steps
1. Clone the repository:
 ```bash
 git clone https://github.com/your-username/ai-shipping-assistant.git
 cd ai-shipping-assistant
   ```
2.Install dependencies:

  ```bash
pip install fastapi uvicorn transformers torch python-dotenv
```

3.Run the application:

```bash
python app.py
```
4.Access the API:
```
Open your browser and go to http://127.0.0.1:8000/docs to use the Swagger UI.
```
## Usage
### Shipping Cost Prediction
Use the /predict-shipping endpoint to get the best shipping option for a given weight and distance.

Example Request:

```
GET /predict-shipping?weight=2&distance=500
```
Example Response:
```
json

{
  "carrier": "FedEx",
  "estimated_cost": 15
}
```
Example:

![image](https://github.com/user-attachments/assets/2dbe551f-a73f-4b31-ac46-ecd16f023127)

## AI Shipping Assistant
Use the /ask-ai endpoint to ask natural language questions about shipping.

Example Request:
```
GET /ask-ai?query=What is the cheapest shipping option for a 3kg package traveling 700 km?
```
Example Response:
```
json

{
  "response": "For a 3kg package traveling 700 km, the cheapest shipping option is USPS at $18."
}
```

Example:

![image](https://github.com/user-attachments/assets/828b6dfa-a314-477b-9b40-16995aec5c85)

## Technical Stack
**Python:** Backend logic.

**FastAPI:** REST API framework.

**Falcon LLM:** Language model for natural language processing.

**Transformers:** Library for loading and interacting with the Falcon model.

**Regex:** For parsing user queries.

## Demo
Check out the video demo [here](https://drive.google.com/file/d/1DeVLZ4qHwj5qIVq7zlOSTpnnmUuKlXjX/view?usp=drive_link).
