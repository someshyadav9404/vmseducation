import requests
import google.generativeai as genai
import json

# --- CONFIGURATION ---
GEMINI_API_KEY = "AIzaSyCY0UJj2iAnGirzqqRAMf8vhnFWyIb8g2w"

# --- SETUP GEMINI ---
genai.configure(api_key=GEMINI_API_KEY)

# --- TOOL: Get City Coordinates (using Open-Meteo geocoding) ---
def get_city_coords(city):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
    resp = requests.get(url)
    data = resp.json()
    if "results" not in data or not data["results"]:
        return {"error": "City not found"}
    coords = data["results"][0]
    return {"lat": coords["latitude"], "lon": coords["longitude"], "name": coords["name"]}

# --- TOOL: Get Weather ---
def get_weather(city):
    coords = get_city_coords(city)
    if "error" in coords:
        return coords
    lat, lon = coords["lat"], coords["lon"]
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&current_weather=true&hourly=precipitation,temperature_2m,weathercode"
    )
    resp = requests.get(url)
    data = resp.json()
    if "current_weather" not in data:
        return {"error": "Weather data not found"}
    weather = data["current_weather"]
    # Weather code mapping (simplified)
    weather_codes = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Drizzle",
        55: "Dense drizzle", 56: "Freezing drizzle", 57: "Dense freezing drizzle",
        61: "Slight rain", 63: "Rain", 65: "Heavy rain", 66: "Freezing rain",
        67: "Heavy freezing rain", 71: "Slight snow", 73: "Snow", 75: "Heavy snow",
        77: "Snow grains", 80: "Slight rain showers", 81: "Rain showers",
        82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers",
        95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Heavy thunderstorm with hail"
    }
    desc = weather_codes.get(weather["weathercode"], "Unknown")
    return {
        "city": coords["name"],
        "temp": weather["temperature"],
        "wind": weather["windspeed"],
        "desc": desc,
        "time": weather["time"]
    }

# --- TOOL: Save Report ---
def save_report(filename, content):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Saved to {filename}."

# --- SYSTEM INSTRUCTION FOR GEMINI ---
SYSTEM_INSTRUCTION = """
You are a real-time Weather Alert & Recommendation Agent for India.
1. Ask the user for the city name.
2. Use `get_weather` to fetch current weather.
3. Summarize the weather and flag any severe conditions (e.g., heavy rain, storms, extreme heat).
4. Give actionable recommendations (e.g., carry umbrella, avoid outdoor activity, health tips).
5. Save the report using `save_report`.
"""

# --- GEMINI MODEL SETUP ---
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    tools=[get_weather, save_report],
    system_instruction=SYSTEM_INSTRUCTION,
)

# --- AGENTIC WORKFLOW ---
def run_agent():
    print("--- Real-Time Weather Alert & Recommendation Agent ---")
    chat = model.start_chat(enable_automatic_function_calling=True)
    city = input("Enter your city name: ").strip()
    user_goal = f"Give me the current weather and recommendations for {city}."
    response = chat.send_message(user_goal)
    print("\n--- Weather Report & Recommendations ---\n")
    print(response.text)

if __name__ == "__main__":
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        print("Please set your Gemini API key.")
    else:
        run_agent()
