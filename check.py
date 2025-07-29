# import google.generativeai as genai

# genai.configure(api_key="AIzaSyAlajdo2sOMdjhZFQJqZWA2-65-QrRyFMA")

# model = genai.GenerativeModel(model_name="models/gemini-pro")

# response = model.generate_content("What is grape black rot and how to treat it?")
# print(response.text)

# import google.generativeai as genai

# genai.configure(api_key="AIzaSyAfQFR9K-FennygQwD0Er77d30JheNhZEI")

# model = genai.GenerativeModel('gemini-pro')

# response = model.generate_content("Say hello, Gemini!")
# print("RESPONSE FROM GEMINI:", response.text)

# import google.generativeai as genai

# genai.configure(api_key="AIzaSyAfQFR9K-FennygQwD0Er77d30JheNhZEI")

# models = genai.list_models()
# for m in models:
#     print(m.name)

import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel("models/gemini-1.5-flash")

response = model.generate_content("What is grape black rot and how to treat it?")
print(response.text)
