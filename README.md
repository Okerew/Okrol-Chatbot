# Okrol-Chatbot
A chatbot created on the base of okrol model by me
![Screenshot 2024-05-13 at 20 15 57](https://github.com/Okerew/Okrol-Chatbot/assets/93822247/d3e1240c-0d7f-4114-8cce-5fefd05d0bb2)
______________________
Installation
---------------
To install firstly download requirements --> `pip install -r requirements`
<br>
Then download spacy en_core_web_sm module --> `python -m spacy download en_core_web_sm`
______________________
Usage
-------------------
The model is not really great and shoudln't be used for anything else than experimenting:
1. It losses some data from epoches
2. It is unstable
3. It bugs out very often
_____________________
How to add data
-------------------
This is a blueprint for example data
```
[
  {
    "text": "Who is your creator?",
    "response": "Okerew is my creator"
  },
  {
    "text": "Hey, hey",
    "response": "Hello"
  }
]
```
